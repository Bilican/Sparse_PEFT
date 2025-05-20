import logging
import math
import os
import time
import json
import shutil
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

class TorchTracemalloc:
    """Utility for monitoring GPU memory usage"""
    def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator
        self.gpu_utilization = []
        self.timestamps = []
        self.start_time = None
        self.log_frequency = 50  # Only log every 50 steps
        self.enabled = True  # Whether memory tracking is enabled

    def __enter__(self):
        if torch.cuda.is_available() and self.enabled:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available() and self.enabled:
            self.gpu_utilization.append(torch.cuda.max_memory_allocated() / (1024 ** 3))
            self.timestamps.append(time.time() - self.start_time)
            # Only log final memory usage, no detailed reporting
            logger.info(f"Final GPU memory peak: {self.gpu_utilization[-1]:.2f} GB")
            
    def log_memory_report(self, step: int) -> Dict:
        """Log detailed memory usage information and return as a dict"""
        if not torch.cuda.is_available() or not self.enabled:
            return {}
            
        # Only log memory at sparse intervals
        if step % self.log_frequency != 0:
            return {}
            
        current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        mem_report = {
            "step": step,
            "current_memory_GB": current_mem,
            "peak_memory_GB": peak_mem,
            "timestamp": time.time() - self.start_time if self.start_time else 0
        }
        
        logger.info(f"Memory at step {step}: current={current_mem:.2f}GB, peak={peak_mem:.2f}GB")
        return mem_report

def setup_optimizer(
    unet: UNet2DConditionModel,
    config: Dict,
    text_encoder_1: Optional[CLIPTextModel] = None,
    text_encoder_2: Optional[CLIPTextModelWithProjection] = None,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Set up the optimizer and learning rate scheduler.
    
    Args:
        unet: The UNet model with adapters
        config: The training configuration
        text_encoder_1: Optional first text encoder for training
        text_encoder_2: Optional second text encoder for training
        
    Returns:
        Tuple of (optimizer, lr_scheduler)
    """
    # Get only the trainable parameters (adapter parameters)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    
    # If we're training text encoders, include those parameters
    if text_encoder_1 is not None and config.get("train_text_encoder", False):
        trainable_text_encoder_params = [p for p in text_encoder_1.parameters() if p.requires_grad]
        # Use different learning rate if specified
        text_encoder_lr = config.get("text_encoder_lr", config.get("learning_rate", 1e-5))
        params_to_optimize = [
            {"params": trainable_params, "lr": config.get("learning_rate", 1e-5)},
            {
                "params": trainable_text_encoder_params, 
                "lr": text_encoder_lr,
                "weight_decay": config.get("adam_weight_decay_text_encoder", 1e-3)
            }
        ]
        if text_encoder_2 is not None:
            trainable_text_encoder_2_params = [p for p in text_encoder_2.parameters() if p.requires_grad]
            params_to_optimize.append({
                "params": trainable_text_encoder_2_params, 
                "lr": text_encoder_lr,
                "weight_decay": config.get("adam_weight_decay_text_encoder", 1e-3)
            })
        
        logger.info(f"Training text encoder(s) with learning rate: {text_encoder_lr}")
    else:
        # Only train the UNet adapter parameters
        params_to_optimize = [{"params": trainable_params, "lr": config.get("learning_rate", 1e-5)}]
    
    # Check if we're using 8-bit Adam
    if config.get("use_8bit_adam", False):
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params_to_optimize,
                lr=config.get("learning_rate", 1e-5),
                betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                weight_decay=config.get("adam_weight_decay", 1e-4),
                eps=config.get("adam_epsilon", 1e-8),
            )
            logger.info("Using 8-bit AdamW optimizer")
        except ImportError:
            logger.warning("bitsandbytes not found - falling back to regular AdamW")
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=config.get("learning_rate", 1e-5),
                betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                weight_decay=config.get("adam_weight_decay", 1e-4),
                eps=config.get("adam_epsilon", 1e-8),
            )
            logger.info("Using adamw optimizer with 8-bit disabled")
    # Check if we're using Prodigy optimizer
    elif config.get("optimizer", "AdamW").lower() == "prodigy":
        try:
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                params_to_optimize,
                lr=config.get("learning_rate", 1e-5),
                betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                beta3=config.get("prodigy_beta3", 0.0),
                weight_decay=config.get("adam_weight_decay", 1e-4),
                eps=config.get("adam_epsilon", 1e-8),
                decouple=config.get("prodigy_decouple", True),
                use_bias_correction=config.get("prodigy_use_bias_correction", True),
                safeguard_warmup=config.get("prodigy_safeguard_warmup", True),
            )
            logger.info("Using Prodigy optimizer")
        except ImportError:
            logger.warning("prodigyopt not found - falling back to regular AdamW")
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=config.get("learning_rate", 1e-5),
                betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
                weight_decay=config.get("adam_weight_decay", 1e-4),
                eps=config.get("adam_epsilon", 1e-8),
            )
            logger.info("Using adamw optimizer with prodigy disabled")
    else:
        # Regular AdamW optimizer
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=config.get("learning_rate", 1e-5),
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
            weight_decay=config.get("adam_weight_decay", 1e-4),
            eps=config.get("adam_epsilon", 1e-8),
            foreach=True,  # Faster implementation
        )
        param_group_info = [{"group": i, "lr": g["lr"], "params": len(g["params"])} for i, g in enumerate(optimizer.param_groups)]
        logger.info(f"Using adamw optimizer with {len(params_to_optimize)} parameter groups")
        
    # Set up learning rate scheduler
    lr_scheduler = create_lr_scheduler(optimizer, config)
    return optimizer, lr_scheduler

def create_lr_scheduler(optimizer, config):
    """Create a learning rate scheduler based on configuration"""
    lr_scheduler_type = config.get("lr_scheduler", "constant")
    lr_scheduler_kwargs = {
        "optimizer": optimizer,
        "num_warmup_steps": config.get("lr_warmup_steps", 0),
    }
    
    # Add specific parameters for different scheduler types
    if lr_scheduler_type != "constant":
        lr_scheduler_kwargs["num_training_steps"] = config.get("max_train_steps", 500)
        
    if lr_scheduler_type == "cosine_with_restarts":
        lr_scheduler_kwargs["num_cycles"] = config.get("lr_num_cycles", 1)
    
    if lr_scheduler_type == "polynomial":
        lr_scheduler_kwargs["power"] = config.get("lr_power", 1.0)
    
    # Create scheduler
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        **lr_scheduler_kwargs
    )
    
    logger.info(f"Using {lr_scheduler_type} learning rate scheduler")
    return lr_scheduler

def get_noise_scheduler_function(
    scheduler: DDPMScheduler,
    config: Dict,
):
    """
    Get the appropriate noise scheduler function based on configuration.
    
    Args:
        scheduler: The noise scheduler
        config: The training configuration
        
    Returns:
        A function that handles noise scheduling and loss calculation
    """
    # Check if we're using the SNR weighting from the "Improved Diffusion" paper
    snr_gamma = config.get("snr_gamma", None)
    
    if snr_gamma is not None:
        logger.info(f"Using SNR weighting with gamma={snr_gamma}")
        
        # Function for computing SNR-weighted loss
        def compute_snr_weight(timesteps, scheduler):
            # Ensure timesteps is the correct type
            if not isinstance(timesteps, torch.Tensor):
                timesteps = torch.tensor(timesteps, dtype=torch.float32)
            elif timesteps.dtype != torch.float32:
                timesteps = timesteps.to(dtype=torch.float32)
                
            # Get alphas_cumprod
            alphas_cumprod = scheduler.alphas_cumprod
            sqrt_alphas_cumprod = alphas_cumprod ** 0.5
            sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5
            
            # Standard formula for the SNR
            snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
            
            # Computing min_snr weights
            gamma = torch.full_like(snr, snr_gamma)
            snr_weight = torch.minimum(snr, gamma) / gamma
            
            return snr_weight
        
        def noise_scheduler_function(model_output, noise, timesteps, scheduler, latents):
            # Compute SNR weights for the timesteps
            weights = compute_snr_weight(timesteps, scheduler).to(model_output.device)
            
            # Expand dimensions to match model output
            weights = weights.view(model_output.shape[0], *([1] * (len(model_output.shape) - 1)))
            
            # Compute weighted MSE loss
            loss = F.mse_loss(model_output, noise, reduction="none")
            loss = (loss * weights).mean()
            
            return loss
    else:
        # Default MSE loss without SNR weighting
        def noise_scheduler_function(model_output, noise, timesteps, scheduler, latents):
            return F.mse_loss(model_output, noise, reduction="mean")
    
    return noise_scheduler_function

def train_one_step(
    batch: Dict,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    noise_scheduler: DDPMScheduler, 
    scheduler_function,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    config: Dict,
    tokenizer_1=None,
    tokenizer_2=None,
) -> torch.Tensor:
    """
    Execute one training step for DreamBooth.
    
    Args:
        batch: The batch of data
        unet: The UNet model with adapters
        vae: The VAE model
        text_encoder_1: The first text encoder
        text_encoder_2: The second text encoder
        noise_scheduler: The noise scheduler
        scheduler_function: Function for noise scheduling and loss calculation
        accelerator: The Accelerator instance
        weight_dtype: The data type for weights
        config: The training configuration
        tokenizer_1: The first tokenizer
        tokenizer_2: The second tokenizer
        
    Returns:
        loss: The loss value for this step
    """
    # Move images to device and get batch size - don't specify dtype
    pixel_values = batch["pixel_values"].to(accelerator.device)
    batch_size = pixel_values.shape[0]
    
    # Get text embeddings
    with torch.no_grad():
        # Tokenize prompts
        if "input_ids_1" not in batch and "prompts" in batch:
            # Tokenize the prompts
            prompts = batch["prompts"]
            text_inputs_1 = tokenizer_1(
                prompts, 
                padding="max_length", 
                max_length=tokenizer_1.model_max_length, 
                truncation=True,
                return_tensors="pt"
            )
            text_inputs_2 = tokenizer_2(
                prompts, 
                padding="max_length", 
                max_length=tokenizer_2.model_max_length, 
                truncation=True,
                return_tensors="pt"
            )
            input_ids_1 = text_inputs_1.input_ids.to(accelerator.device)
            input_ids_2 = text_inputs_2.input_ids.to(accelerator.device)
        else:
            # If the batch already contains tokenized inputs, use them
            input_ids_1 = batch["input_ids_1"].to(accelerator.device)
            input_ids_2 = batch["input_ids_2"].to(accelerator.device)
        
        # Get embeddings from text encoders - don't specify dtype
        encoder_hidden_states_1 = text_encoder_1(input_ids_1)[0]
        encoder_hidden_states_2 = text_encoder_2(input_ids_2)[0]
        
        # For SDXL, text_encoder_2 also returns pooled output (dim=2), 
        # but we need the sequence output (dim=3 like encoder_hidden_states_1)
        # So either get the hidden states directly or ensure same dimensions
        if encoder_hidden_states_2.ndim != encoder_hidden_states_1.ndim:
            # If we got pooled output, we need to get the hidden states from text_encoder_2 instead
            # This is important for SDXL pipeline
            text_encoder_2_output = text_encoder_2(input_ids_2, output_hidden_states=True)
            encoder_hidden_states_2 = text_encoder_2_output.hidden_states[-2]
            
        # Concatenate the output of both encoders
        encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)
        
        # Create SDXL-specific conditioning for the UNet
        # SDXL expects time_ids for conditioning on original size and crop info
        # Generate default values if not provided in the batch
        original_size = batch.get("original_sizes", [(1024, 1024)] * batch_size)
        crop_top_left = batch.get("crop_top_lefts", [(0, 0)] * batch_size)
        
        # SDXL requires a specific format for time_ids
        target_size = (config.get("resolution", 1024), config.get("resolution", 1024))
        time_ids = []
        for orig_size, crop_tl in zip(original_size, crop_top_left):
            # SDXL time_ids contains: [orig_width, orig_height, crop_top, crop_left, target_width, target_height]
            time_id = [
                orig_size[0],  # original width
                orig_size[1],  # original height
                crop_tl[0],    # crop top
                crop_tl[1],    # crop left
                target_size[0],  # target width
                target_size[1],  # target height
            ]
            time_ids.append(time_id)
        
        # Convert to tensor and send to device - don't specify dtype
        time_ids = torch.tensor(time_ids, device=accelerator.device)
        
        # Check if we have pooled embeddings for text_encoder_2
        pooled_output = None
        if hasattr(text_encoder_2, "text_projection"):
            # Get pooled output for SDXL
            pooled_output = text_encoder_2(input_ids_2, output_hidden_states=False).text_embeds
        
        # Get SDXL-specific conditioning
        added_cond_kwargs = {
            "text_embeds": pooled_output,
            "time_ids": time_ids,
        }
    
    # Encode images to latent space with VAE
    with torch.no_grad():
        # Ensure pixel_values are in the same dtype as the VAE
        pixel_values_dtype = pixel_values.to(dtype=torch.float32)
        latents = vae.encode(pixel_values_dtype).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        latents = latents.to(dtype=weight_dtype)

    
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    
    # Sample a random timestep for each image - IMPORTANT: timesteps must be long
    timesteps = torch.randint(
        0, 
        noise_scheduler.config.num_train_timesteps, 
        (batch_size,), 
        device=latents.device
    ).long()
    
    # Add noise to the latents according to the noise magnitude at each timestep
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Run the UNet model forward
    model_output = unet(
        noisy_latents,
        timesteps,  # Must be long for embedding layers
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
    ).sample
    
    # Convert all tensors to float32 for stable gradient computation
    model_output = model_output.float()
    noise = noise.float()
    latents = latents.float()
    
    # Compute loss
    loss = scheduler_function(model_output, noise, timesteps, noise_scheduler, latents)
    
    # Check for NaN and replace with zero if needed
    if torch.isnan(loss).any():
        logger.warning("NaN detected in loss! Using small non-zero loss instead.")
        # Use a small non-zero loss value that requires grad
        loss = torch.tensor(1e-5, device=loss.device, dtype=loss.dtype, requires_grad=True)
    
    # Handle prior preservation if enabled
    if config.get("with_prior_preservation", False) and "class_images" in batch:
        # Calculate a separate loss for the class images and add it to the instance loss
        class_model_output = model_output[batch_size//2:]
        class_noise = noise[batch_size//2:]
        class_timesteps = timesteps[batch_size//2:]
        class_latents = latents[batch_size//2:]
        
        prior_loss = scheduler_function(
            class_model_output, 
            class_noise, 
            class_timesteps,
            noise_scheduler,
            class_latents
        )
        
        # Combine losses with a weighting coefficient
        prior_loss_weight = config.get("prior_loss_weight", 1.0)
        loss = loss + prior_loss_weight * prior_loss
            
    return loss

def train_loop(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer_1,
    tokenizer_2,
    noise_scheduler: DDPMScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    config: Dict,
    validation_function: Optional[Callable] = None,
) -> None:
    """
    Execute the main training loop for DreamBooth fine-tuning.
    
    Args:
        unet: The UNet model with adapters
        vae: The VAE model
        text_encoder_1: The first text encoder
        text_encoder_2: The second text encoder
        tokenizer_1: The first tokenizer
        tokenizer_2: The second tokenizer
        noise_scheduler: The noise scheduler
        train_dataloader: The dataloader for training data
        optimizer: The optimizer
        lr_scheduler: The learning rate scheduler
        accelerator: The Accelerator instance
        weight_dtype: The data type for inputs (fp16/bf16 for mixed precision)
        config: The training configuration
        validation_function: Optional function for validation during training
    """
    # Get the noise scheduler function for loss calculation
    scheduler_function = get_noise_scheduler_function(noise_scheduler, config)
    
    # Set up training parameters
    max_train_steps = config.get("max_train_steps", 500)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    checkpointing_steps = config.get("checkpointing_steps", 500)
    validation_steps = config.get("validation_steps", 100)
    checkpoints_total_limit = config.get("checkpoints_total_limit", None)
    memory_logging_steps = config.get("memory_logging_steps", 500)  # Very sparse memory logging, every 500 steps by default
    max_grad_norm = config.get("max_grad_norm", 1.0)  # Get max grad norm from config, default to 1.0
    
    # Check if memory measurement is enabled
    measure_memory_usage = config.get("measure_memory_usage", False)
    
    # Set the random seed for reproducibility
    if config.get("seed", None) is not None:
        set_seed(config.get("seed"))
    
    # Prepare everything with accelerator
    # Note: We're ensuring the models stay in float32 but can handle lower precision inputs
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Double-check that the UNet parameters are in float32 to avoid FP16 gradient issues
    for param in unet.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    
    # Calculate the total number of steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    # Start training
    total_batch_size = (
        config.get("train_batch_size", 1) * 
        accelerator.num_processes * 
        gradient_accumulation_steps
    )
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches per epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.get('train_batch_size', 1)}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Only show the progress bar once on each machine
    progress_bar = tqdm(
        range(max_train_steps), 
        disable=not accelerator.is_local_main_process,
        desc="Training steps"
    )
    global_step = 0
    best_validation_loss = float('inf')
    
    # Create memory tracker
    memtracker = TorchTracemalloc(accelerator)
    memtracker.enabled = measure_memory_usage
    
    # Create metrics dictionary to track training metrics
    training_metrics = {
        "loss": [],
        "lr": [],
        "memory_metrics": [],
    }
    
    # Simple AMP-like scaling for manual mixed precision
    # This is a simple alternative to torch.cuda.amp.GradScaler since we're not using accelerator's mixed precision
    use_scaling = weight_dtype != torch.float32
    loss_scale = 2**16 if use_scaling else 1.0
    
    # Skip automatic full model checkpointing at the end of training
    # We'll save only the adapter weights separately
    if hasattr(accelerator, 'save_state'):
        original_save_state = accelerator.save_state
        # Replace with a no-op function temporarily
        accelerator.save_state = lambda *args, **kwargs: logger.info("Skipping automatic full model checkpointing")

    for epoch in range(num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        # Set up memory tracking
        with memtracker:
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if config.get("resume_from_checkpoint", None) is not None and epoch == 0:
                    if global_step < config.get("resume_step", 0):
                        global_step += 1
                        continue
                
                with accelerator.accumulate(unet):
                    # Forward and backward pass
                    loss = train_one_step(
                        batch=batch,
                        unet=unet,
                        vae=vae,
                        text_encoder_1=text_encoder_1,
                        text_encoder_2=text_encoder_2,
                        noise_scheduler=noise_scheduler,
                        scheduler_function=scheduler_function,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        config=config,
                        tokenizer_1=tokenizer_1,
                        tokenizer_2=tokenizer_2,
                    )
                    
                    # Gather the losses from all processes
                    avg_loss = accelerator.gather(loss.repeat(config.get("train_batch_size", 1))).mean()
                    train_loss += avg_loss.item()
                    
                    # Backpropagate
                    # The train_one_step function already ensures loss is in float32
                    accelerator.backward(loss)
                    
                    # Apply gradient clipping
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    
                    # Update parameters
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    # Log metrics
                    if global_step % config.get("logging_steps", 10) == 0:
                        current_loss = loss.detach().item()
                        current_lr = lr_scheduler.get_last_lr()[0]
                        logs = {"loss": current_loss, "lr": current_lr}
                        
                        # Track metrics
                        training_metrics["loss"].append((global_step, current_loss))
                        training_metrics["lr"].append((global_step, current_lr))
                        
                        # Log memory if available, but only at sparse intervals defined by memory_logging_steps
                        if torch.cuda.is_available() and measure_memory_usage and global_step % memory_logging_steps == 0:
                            memory_metrics = memtracker.log_memory_report(global_step)
                            training_metrics["memory_metrics"].append(memory_metrics)
                        
                        accelerator.log(logs, step=global_step)
                        progress_bar.set_postfix(**logs)
                    
                    # Run validation if needed
                    if validation_function is not None and global_step % validation_steps == 0:
                        logger.info(f"Running validation at step {global_step}")
                        validation_loss, validation_images = validation_function(
                            unet=accelerator.unwrap_model(unet),
                            global_step=global_step,
                        )
                        
                        is_best = False
                        if validation_loss < best_validation_loss:
                            best_validation_loss = validation_loss
                            is_best = True
                            logger.info(f"New best validation loss: {best_validation_loss:.6f}")
                        
                        # Log validation metrics
                        accelerator.log({"validation_loss": validation_loss}, step=global_step)
                        
                        # Save checkpoint with best model indication if it's the best so far
                        output_dir = os.path.join(
                            config.get("output_dir", "output"), 
                            f"checkpoint-{global_step}"
                        )
                        save_checkpoint(
                            accelerator, unet, optimizer, lr_scheduler, 
                            output_dir, config, validation_loss, is_best
                        )
                    
                    # Save regular checkpoint
                    elif global_step % checkpointing_steps == 0:
                        output_dir = os.path.join(
                            config.get("output_dir", "output"), 
                            f"checkpoint-{global_step}"
                        )
                        save_checkpoint(accelerator, unet, optimizer, lr_scheduler, output_dir, config)
                        
                        # Manage total number of checkpoints if specified
                        if checkpoints_total_limit is not None:
                            cleanup_checkpoints(
                                config.get("output_dir", "output"), 
                                checkpoints_total_limit
                            )
                
                if global_step >= max_train_steps:
                    break
        
        # Calculate average loss over the epoch
        train_loss = train_loss / (len(train_dataloader) * gradient_accumulation_steps)
        logger.info(f"Epoch {epoch}: Average loss: {train_loss}")
    
    # Save final checkpoint
    output_dir = os.path.join(config.get("output_dir", "output"), "final_model")
    save_checkpoint(accelerator, unet, optimizer, lr_scheduler, output_dir, config)
    
    # Save training metrics
    if accelerator.is_main_process:
        metrics_path = os.path.join(config.get("output_dir", "output"), "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(training_metrics, f)
        logger.info(f"Training metrics saved to {metrics_path}")
    
    # Restore original save_state function
    if hasattr(accelerator, 'save_state') and 'original_save_state' in locals():
        accelerator.save_state = original_save_state
        
    logger.info("Training complete!")

def save_checkpoint(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: str,
    config: Dict,
    validation_loss: Optional[float] = None,
    is_best: bool = False,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        accelerator: The Accelerator instance
        unet: The UNet model with adapters
        optimizer: The optimizer
        lr_scheduler: The learning rate scheduler
        output_dir: Output directory for the checkpoint
        config: The training configuration
        validation_loss: Optional validation loss for this checkpoint
        is_best: Whether this checkpoint is the best so far
    """
    logger.info(f"Saving checkpoint to {output_dir}")
    accelerator.save_state(output_dir)
    
    # Save adapter weights separately for easy loading
    if accelerator.is_main_process:
        # Get unwrapped model
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        # Create directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save adapter state
        if hasattr(unwrapped_unet, "save_adapter"):
            adapter_path = os.path.join(output_dir, "adapter")
            unwrapped_unet.save_adapter(adapter_path, "default")  # Assuming the adapter name is "default"
            logger.info(f"Adapter weights saved to {adapter_path}")
            
            # Save additional metadata about the checkpoint
            metadata = {
                "adapter_type": config.get("adapter_type", "loha"),
                "timestamp": time.time(),
                "steps": config.get("global_step", 0),
            }
            
            if validation_loss is not None:
                metadata["validation_loss"] = validation_loss
                
            with open(os.path.join(adapter_path, "adapter_metadata.json"), "w") as f:
                json.dump(metadata, f)
        
        # Save in safetensors format if requested
        if config.get("save_safetensors", True) and hasattr(unwrapped_unet, "get_adapter"):
            try:
                adapter_state_dict = unwrapped_unet.get_adapter("default")
                if adapter_state_dict:
                    safetensors_path = os.path.join(output_dir, "adapter_model.safetensors")
                    save_file(adapter_state_dict, safetensors_path)
                    logger.info(f"Adapter weights saved in safetensors format to {safetensors_path}")
            except Exception as e:
                logger.warning(f"Failed to save adapter in safetensors format: {e}")
        
        # Save training args
        with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
            import yaml
            yaml.dump(config, f)
        
        logger.info(f"Configuration saved to {os.path.join(output_dir, 'training_config.yaml')}")
        
        # If this is the best model, save a copy or symlink
        if is_best:
            best_model_dir = os.path.join(os.path.dirname(output_dir), "best_model")
            
            # Remove existing best model directory if it exists
            if os.path.exists(best_model_dir):
                if os.path.islink(best_model_dir):
                    os.unlink(best_model_dir)
                else:
                    shutil.rmtree(best_model_dir)
            
            # Create a symbolic link to the best checkpoint
            if config.get("symlink_best_model", True):
                os.symlink(os.path.basename(output_dir), best_model_dir)
                logger.info(f"Created symlink to best model: {best_model_dir} -> {output_dir}")
            else:
                # Copy instead of symlink
                shutil.copytree(output_dir, best_model_dir)
                logger.info(f"Copied best model to {best_model_dir}")

def cleanup_checkpoints(output_dir: str, checkpoints_total_limit: int) -> None:
    """
    Cleanup old checkpoints to save disk space, keeping only the most recent ones.
    
    Args:
        output_dir: Base output directory containing checkpoints
        checkpoints_total_limit: Maximum number of checkpoints to keep
    """
    if not os.path.exists(output_dir):
        return
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if len(checkpoints) <= checkpoints_total_limit:
        return
    
    # Sort checkpoints by step number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    
    # Remove oldest checkpoints, keeping only the most recent ones
    for checkpoint in checkpoints[:-checkpoints_total_limit]:
        checkpoint_path = os.path.join(output_dir, checkpoint)
        logger.info(f"Removing old checkpoint: {checkpoint_path}")
        shutil.rmtree(checkpoint_path)

def load_checkpoint(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_dir: str,
) -> Tuple[UNet2DConditionModel, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
    """
    Load a training checkpoint.
    
    Args:
        accelerator: The Accelerator instance
        unet: The UNet model with adapters
        optimizer: The optimizer
        lr_scheduler: The learning rate scheduler
        checkpoint_dir: Directory containing the checkpoint
        
    Returns:
        Tuple of (UNet model, optimizer, lr_scheduler, global_step)
    """
    logger.info(f"Loading checkpoint from {checkpoint_dir}")
    
    # Check if the checkpoint exists
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
    
    # Load the checkpoint state
    accelerator.load_state(checkpoint_dir)
    
    # Get the global step from the checkpoint directory name
    if os.path.basename(checkpoint_dir).startswith("checkpoint-"):
        global_step = int(os.path.basename(checkpoint_dir).split("-")[1])
    else:
        # Try to get the step from adapter metadata
        metadata_path = os.path.join(checkpoint_dir, "adapter", "adapter_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                global_step = metadata.get("steps", 0)
        else:
            logger.warning(f"Could not determine global step from checkpoint {checkpoint_dir}")
            global_step = 0
    
    logger.info(f"Checkpoint loaded, resuming from global step {global_step}")
    return unet, optimizer, lr_scheduler, global_step 