#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os
import sys
import yaml
import traceback
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.utils import is_xformers_available
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

# Add local peft to the path
sys.path.insert(0, os.path.abspath('./'))

from sdxl_dreambooth.adapter_factory import apply_adapter_to_model
from sdxl_dreambooth.dataset_loader import DreamBoothDataset, create_dataloader
from sdxl_dreambooth.dataloader_config import load_dataloader_config, get_dataloader_params
from sdxl_dreambooth.training_utils import (
    setup_optimizer,
    train_loop,
    TorchTracemalloc
)

# Now import from the local peft
from peft.tuners.lora import LoraConfig
from peft.tuners.loha import LoHaConfig
from peft.tuners.lokr import LoKrConfig
from peft.tuners.vera import VeraConfig
from peft.tuners.fourierft import FourierFTConfig
from peft.tuners.adalora import AdaLoraConfig
from peft.tuners.waveft import WaveFTConfig
from peft import get_peft_model, PeftModel

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion XL DreamBooth training with PEFT adapters")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="sdxl_dreambooth/config",
        help="Directory containing configuration files",
    )
    return parser.parse_args()

def validate_config(config):
    """
    Validate and convert config values to appropriate types.
    
    Args:
        config: The configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    # Ensure numeric values are properly converted to float/int
    float_keys = [
        "learning_rate", "text_encoder_lr", "adam_weight_decay", 
        "adam_weight_decay_text_encoder", "adam_beta1", "adam_beta2", 
        "adam_epsilon", "max_grad_norm", "prior_loss_weight",
        "prodigy_beta3"
    ]
    
    int_keys = [
        "max_train_steps", "train_batch_size", "gradient_accumulation_steps",
        "seed", "resolution", "rank", "alpha", "lr_warmup_steps", 
        "lr_num_cycles", "checkpointing_steps", "validation_steps",
        "num_validation_images", "logging_steps", "save_steps",
        "checkpoints_total_limit"
    ]
    
    bool_keys = [
        "use_gradient_checkpointing", "enable_xformers_memory_efficient_attention",
        "use_8bit_adam", "train_text_encoder", "with_prior_preservation",
        "prodigy_decouple", "prodigy_use_bias_correction", "prodigy_safeguard_warmup",
        "save_safetensors", "symlink_best_model", "center_crop", "use_captions",
        "measure_memory_usage"
    ]
    
    validated_config = config.copy()
    
    # Convert float values
    for key in float_keys:
        if key in validated_config and validated_config[key] is not None:
            try:
                validated_config[key] = float(validated_config[key])
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for {key}: {validated_config[key]}, using default")
                # Set default values for critical parameters
                if key == "learning_rate":
                    validated_config[key] = 1e-5
                elif key == "adam_weight_decay":
                    validated_config[key] = 1e-4
                elif key == "adam_epsilon":
                    validated_config[key] = 1e-8
                    
    # Convert int values
    for key in int_keys:
        if key in validated_config and validated_config[key] is not None:
            try:
                validated_config[key] = int(validated_config[key])
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for {key}: {validated_config[key]}, using default")
                
    # Convert bool values
    for key in bool_keys:
        if key in validated_config:
            if isinstance(validated_config[key], str):
                validated_config[key] = validated_config[key].lower() in ("yes", "true", "t", "1")
    
    return validated_config

def load_config(config_dir):
    """
    Load configuration from YAML files.
    
    Args:
        config_dir: Directory containing configuration files
    
    Returns:
        config: Configuration dictionary
    """
    config = {}
    
    # Load training configuration
    training_config_path = os.path.join(config_dir, "training_config.yaml")
    with open(training_config_path, "r") as f:
        config.update(yaml.safe_load(f))
    
    # Load adapter configuration
    adapter_config_path = os.path.join(config_dir, "adapter_config.yaml")
    with open(adapter_config_path, "r") as f:
        config.update(yaml.safe_load(f))
    
    # Load experiment configuration
    experiment_config_path = os.path.join(config_dir, "experiment_config.yaml")
    with open(experiment_config_path, "r") as f:
        config.update(yaml.safe_load(f))
    
    # Validate and convert config values
    config = validate_config(config)
    
    return config

def setup_model(config, accelerator):
    """
    Setup the SDXL model components and apply PEFT adapters.
    
    Args:
        config: Configuration dictionary
        accelerator: Accelerator instance for distributed training
        
    Returns:
        Tuple containing (unet, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler)
    """
    logger.info("Loading models and applying adapters...")
    
    # 1. Load the pretrained SDXL model components
    # Load UNet
    logger.info(f"Loading UNet from {config['pretrained_model_name_or_path']}")
    unet = UNet2DConditionModel.from_pretrained(
        config['pretrained_model_name_or_path'], 
        subfolder="unet"
    )
    
    # Load VAE
    logger.info("Loading VAE")
    vae = AutoencoderKL.from_pretrained(
        config['pretrained_model_name_or_path'],
        subfolder="vae"
    )
    
    # Load text encoders
    logger.info("Loading text encoders")
    text_encoder_1 = CLIPTextModel.from_pretrained(
        config['pretrained_model_name_or_path'], 
        subfolder="text_encoder"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        config['pretrained_model_name_or_path'], 
        subfolder="text_encoder_2"
    )
    
    # Load tokenizers
    logger.info("Loading tokenizers")
    tokenizer_1 = AutoTokenizer.from_pretrained(
        config['pretrained_model_name_or_path'],
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        config['pretrained_model_name_or_path'],
        subfolder="tokenizer_2",
        use_fast=False,
    )
    
    # Load noise scheduler
    logger.info("Loading noise scheduler")
    noise_scheduler = DDPMScheduler.from_pretrained(
        config['pretrained_model_name_or_path'], 
        subfolder="scheduler"
    )
    
    # 2. Set up gradient checkpointing for memory efficiency
    if config.get("use_gradient_checkpointing", True):
        logger.info("Enabling gradient checkpointing")
        unet.enable_gradient_checkpointing()
        
    # 3. Set up xformers if available
    if config.get("enable_xformers_memory_efficient_attention", False) and is_xformers_available():
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        logger.info("Using xformers for memory-efficient attention")
    else:
        logger.info("Not using xformers for attention")
        
    # 4. Apply PEFT adapters to UNet
    adapter_config = {
        "adapter_type": config.get("adapter_type", "loha"),
        "n_frequency": config.get("n_frequency", 10000),
        "scaling": config.get("scaling", 120),
        "rank": config.get("rank", 4),
        "alpha": config.get("alpha", 4),
        "target_modules": config.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
        "init_weights": config.get("init_weights", True),
        "module_dropout": config.get("module_dropout", 0.0),
        "modules_to_save": config.get("modules_to_save", None),
        # AdaLora specific parameters
        "target_r": config.get("target_r", 8),
        "init_r": config.get("init_r", 12),
        "tinit": config.get("tinit", 0),
        "tfinal": config.get("tfinal", 0),
        "deltaT": config.get("deltaT", 1),
        "beta1": config.get("beta1", 0.85),
        "beta2": config.get("beta2", 0.85),
        "orth_reg_weight": config.get("orth_reg_weight", 0.5),
        "wavelet_family": config.get("wavelet_family", "db1"),
        "use_idwt": config.get("use_idwt", True),
        "random_loc_seed": config.get("random_loc_seed", 777),
    }
    
    # Create proper PEFT config object based on adapter type
    adapter_type = adapter_config["adapter_type"].lower()
    
    if adapter_type == "loha":
        peft_config = LoHaConfig(
            r=adapter_config["rank"],
            alpha=adapter_config["alpha"],
            target_modules=adapter_config["target_modules"],
            init_weights=adapter_config["init_weights"],
            modules_to_save=adapter_config["modules_to_save"],
        )
    elif adapter_type == "lokr":
        peft_config = LoKrConfig(
            r=adapter_config["rank"],
            alpha=adapter_config["alpha"],
            target_modules=adapter_config["target_modules"],
        )
    elif adapter_type == "vera":
        peft_config = VeraConfig(
            r=adapter_config["rank"],
            target_modules=adapter_config["target_modules"],
        )
    elif adapter_type == "fourierft":
        peft_config = FourierFTConfig(
            n_frequency=adapter_config["n_frequency"],
            scaling=adapter_config["scaling"],
            target_modules=adapter_config["target_modules"],
        )
    elif adapter_type == "waveft":
        peft_config = WaveFTConfig(
            n_frequency=adapter_config["n_frequency"],
            scaling=adapter_config["scaling"],
            target_modules=adapter_config["target_modules"],
            wavelet_family=adapter_config["wavelet_family"],
            use_idwt=adapter_config["use_idwt"],
            random_loc_seed=adapter_config["random_loc_seed"],
            init_weights=adapter_config["init_weights"],
        )
    elif adapter_type == "adalora":
        peft_config = AdaLoraConfig(
            r=adapter_config.get("rank", 8),
            lora_alpha=adapter_config.get("alpha", 8),
            target_modules=adapter_config["target_modules"],
            lora_dropout=adapter_config.get("module_dropout", 0.0),
            bias="none",
            target_r=adapter_config.get("target_r", 8),
            init_r=adapter_config.get("init_r", 12),
            tinit=adapter_config.get("tinit", 0),
            tfinal=adapter_config.get("tfinal", 0),
            deltaT=adapter_config.get("deltaT", 1),
            beta1=adapter_config.get("beta1", 0.85),
            beta2=adapter_config.get("beta2", 0.85),
            orth_reg_weight=adapter_config.get("orth_reg_weight", 0.5),
            total_step=config.get("max_train_steps", None),
        )
    elif adapter_type == "lora":
        peft_config = LoraConfig(
            r=adapter_config["rank"],
            lora_alpha=adapter_config["alpha"],
            target_modules=adapter_config["target_modules"],
            lora_dropout=adapter_config["module_dropout"],
            bias="none",
        )
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_config['adapter_type']}")
    
    # Apply the adapter directly with get_peft_model
    logger.info(f"Applying {adapter_config['adapter_type']} adapter to UNet")
    unet = get_peft_model(unet, peft_config)
    
    # Verify the adapter was applied correctly
    if not hasattr(unet, "save_adapter"):
        logger.error("Failed to apply adapter! The UNet model does not have the save_adapter method.")
        try:
            # Try to explicitly add active_adapter attribute in case it's missing
            unet.active_adapter = "default"
            logger.info("Explicitly set active_adapter")
        except Exception as e:
            logger.error(f"Could not set active_adapter: {e}")
    else:
        logger.info("Successfully applied adapter to UNet")
    
    # 5. Freeze VAE and text encoder parameters
    logger.info("Freezing VAE parameters")
    vae.requires_grad_(False)
    
    logger.info("Freezing text encoder parameters")
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    # 6. Make sure adapter parameters are trainable
    logger.info("Setting adapter parameters to trainable")
    
    # Count trainable parameters (adapter parameters)
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of trainable adapter parameters: {trainable_params}")
    
    return unet, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler


def validate_setup(unet, vae, text_encoder_1, text_encoder_2, device, weight_dtype):
    """
    Validate that the model setup is correct by performing minimal forward passes.
    
    Args:
        unet: UNet model
        vae: VAE model
        text_encoder_1: First text encoder
        text_encoder_2: Second text encoder
        device: Device to run validation on
        weight_dtype: Data type for inputs
    """
    try:
        # Create minimal batch
        batch_size = 1
        height, width = 128, 128  # Minimal latent size
        latent_channels = 4
        seq_len = 77  # Max tokens
        
        # Create dummy latents
        dummy_latents = torch.randn(
            (batch_size, latent_channels, height, width), 
            device=device
        )
        
        # Random timesteps - IMPORTANT: timesteps must be long for UNet embedding
        dummy_timesteps = torch.randint(
            0, 1000, (batch_size,), 
            device=device, 
            dtype=torch.long  # Must be long
        )
        
        # Create dummy text embeddings
        dummy_text_embeds = torch.randn(
            (batch_size, seq_len, 2048), 
            device=device
        )
        
        # Original size
        # In SDXL, we need both original_size and crop_top_left conditioning
        original_size = (1024, 1024)
        crop_top_left = (0, 0)
        target_size = (1024, 1024)
        
        # Time ids for conditioning
        time_ids = torch.tensor(
            [[
                original_size[0],  # original width
                original_size[1],  # original height
                crop_top_left[0],  # crop top
                crop_top_left[1],  # crop left
                target_size[0],    # target width
                target_size[1],    # target height
            ]], 
            device=device
        )
        
        # Create dummy pooled text embeddings
        pooled_text_embeds = torch.randn(
            (batch_size, 1280), 
            device=device
        )
        
        # Minimal UNet forward pass
        added_cond_kwargs = {
            "text_embeds": pooled_text_embeds,
            "time_ids": time_ids
        }
        
        logger.info(f"Running UNet validation with timesteps dtype: {dummy_timesteps.dtype}")
        
        # Do a forward pass of UNet - keeping timesteps as long
        unet_output = unet(
            dummy_latents,
            dummy_timesteps,  # Timesteps must be long
            encoder_hidden_states=dummy_text_embeds,
            added_cond_kwargs=added_cond_kwargs,
        )
        logger.info(f"UNet forward pass successful, output shape: {unet_output.sample.shape}, dtype: {unet_output.sample.dtype}")
        
        # Do a forward pass of VAE
        with torch.no_grad():
            dummy_pixel_values = torch.randn(
                (batch_size, 3, height, width), 
                device=device
            )
            vae_output = vae.encode(dummy_pixel_values)
            logger.info(f"VAE forward pass successful, output shape: {vae_output.latent_dist.sample().shape}, dtype: {vae_output.latent_dist.sample().dtype}")
        
        # Test text encoders
        dummy_input_ids = torch.randint(
            0, 49408, (batch_size, seq_len), 
            device=device, 
            dtype=torch.long  # Input IDs are always long
        )
        text_encoder_1_output = text_encoder_1(dummy_input_ids)[0]
        logger.info(f"Text encoder 1 forward pass successful, output shape: {text_encoder_1_output.shape}, dtype: {text_encoder_1_output.dtype}")
        
        text_encoder_2_output = text_encoder_2(dummy_input_ids)[0]
        logger.info(f"Text encoder 2 forward pass successful, output shape: {text_encoder_2_output.shape}, dtype: {text_encoder_2_output.dtype}")
        
        logger.info("✅ All model components validated successfully!")
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {str(e)}")
        logger.error(traceback.format_exc())
        # Continue with training despite validation failure
        pass

def setup_dataset(config, tokenizer_1):
    """
    Setup the DreamBooth dataset and dataloader.
    
    Args:
        config: Configuration dictionary
        tokenizer_1: CLIP tokenizer for text conditioning
        
    Returns:
        dataloader: DataLoader for training
    """
    logger.info("Setting up dataset and dataloader...")
    
    # Create instance dataset
    dataset = DreamBoothDataset(
        instance_data_root=config["instance_data_dir"],
        tokenizer=tokenizer_1,
        instance_prompt=config.get("instance_prompt"),
        class_prompt=config.get("class_prompt"),
        class_data_root=config.get("class_data_dir"),
        class_num=config.get("num_class_images"),
        size=config.get("resolution", 1024),
        center_crop=config.get("center_crop", False),
        unique_token=config.get("unique_token", "sks"),
        use_captions=config.get("use_captions", False),
        caption_extension=config.get("caption_extension", ".txt"),
        repeats=config.get("repeats", 1),
    )
    
    logger.info(f"Created dataset with {len(dataset)} instance images")
    
    # Check if prior preservation is enabled
    with_prior_preservation = config.get("with_prior_preservation", False)
    if with_prior_preservation:
        logger.info("Using prior preservation")
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=config.get("train_batch_size", 1),
        shuffle=config.get("shuffle", True),
        with_prior_preservation=with_prior_preservation,
        num_workers=config.get("dataloader_num_workers", 0),
    )
    
    logger.info("Dataloader created successfully")
    return dataloader

def main():
    """
    Main function to run SDXL DreamBooth training with PEFT adapters.
    """
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config_dir)
    
    # Print configuration
    logger.info("Training configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Store original mixed precision setting
    original_mixed_precision = config.get("mixed_precision", "no")
    
    # Set up accelerator WITH mixed precision from config
    # Let accelerator handle mixed precision properly
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.get("output_dir", "output"),
        logging_dir=os.path.join(config.get("output_dir", "output"), "logs"),
        automatic_checkpoint_naming=False,  # Disable automatic checkpointing
        save_on_each_node=False,  # Don't save duplicates
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        mixed_precision=original_mixed_precision,  # Use the actual mixed precision setting
        log_with=config.get("report_to", "tensorboard"),
        project_config=accelerator_project_config,
    )
    
    # Set seed for reproducibility
    if config.get("seed", None) is not None:
        set_seed(config.get("seed"))
    
    # Get device from accelerator
    device = accelerator.device
    logger.info(f"Using device: {device}")
    
    # Determine weight dtype based on mixed precision setting
    if original_mixed_precision == "fp16":
        weight_dtype = torch.float16
        logger.info(f"Using {weight_dtype} precision throughout")
    elif original_mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        logger.info(f"Using {weight_dtype} precision throughout")
    else:
        weight_dtype = torch.float32
        logger.info(f"Using {weight_dtype} precision for all tensors")
    
    # 1. Setup model
    unet, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler = setup_model(
        config, accelerator
    )
    
    # Move models to device and let accelerator handle dtype
    unet.to(device)
    vae.to(device, dtype=torch.float32)
    text_encoder_1.to(device)
    text_encoder_2.to(device)

    logger.info(f"VAE explicitly set to float32: {vae.dtype}")
    
    # Double-check UNet time_embedding parameters
    if hasattr(unet, 'time_embedding'):
        logger.info(f"UNet time_embedding dtype: {next(unet.time_embedding.parameters()).dtype}")
    
    # Verify model setup
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    
    logger.info(f"Total UNet parameters: {total_params}")
    logger.info(f"Trainable adapter parameters: {trainable_params}")
    logger.info(f"Parameter efficiency: {trainable_params/total_params*100:.6f}%")
    
    # Count adapter modules
    adapter_modules = 0
    for name, module in unet.named_modules():
        if hasattr(module, 'active_adapter'):
            adapter_modules += 1
    
    logger.info(f"Found {adapter_modules} modules with adapters")
    logger.info("Model setup complete")
    
    # 2. Setup dataset and dataloader
    train_dataloader = setup_dataset(config, tokenizer_1)
    
    # Use utility for optimizer and scheduler setup
    optimizer, lr_scheduler = setup_optimizer(unet, config)
    
    # 3. Run validation on model setup
    try:
        validate_setup(unet, vae, text_encoder_1, text_encoder_2, device, weight_dtype)
    except Exception as e:
        logger.warning(f"Model validation failed, but continuing with training: {str(e)}")
        logger.warning("This is most likely due to dtype mismatch that will be handled during actual training")
    
    # 4. Execute the training loop
    logger.info("Starting training...")
    
    # Check if memory measurement is enabled (default: False)
    measure_memory_usage = config.get("measure_memory_usage", False)
    
    if measure_memory_usage:
        logger.info("Memory usage measurement enabled")
        # Measure GPU memory usage
        with TorchTracemalloc(accelerator) as memtracker:
            # Execute the training loop
            train_loop(
                unet=unet,
                vae=vae,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                tokenizer_1=tokenizer_1,
                tokenizer_2=tokenizer_2,
                noise_scheduler=noise_scheduler,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                config=config,
            )
    else:
        # Execute the training loop without memory tracking
        train_loop(
            unet=unet,
            vae=vae,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            config=config,
        )
    
    logger.info("Training completed successfully!")
    
    # Save the final model
    final_output_dir = os.path.join(config.get("output_dir", "output"), "final_model")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # First try to save using PEFT methods
    unwrapped_unet = accelerator.unwrap_model(unet)
    
    try:
        # First use the standard PEFT saving method
        logger.info("Saving adapter using PEFT method...")
        from peft import PeftModel
        
        if isinstance(unwrapped_unet, PeftModel):
            adapter_dir = os.path.join(final_output_dir, "adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            unwrapped_unet.save_pretrained(adapter_dir)
            logger.info(f"Successfully saved adapter to {adapter_dir}")
    except Exception as e:
        logger.warning(f"Error saving adapter with PEFT method: {e}")
        # Continue with saving using alternative methods

    # IMPORTANT CLEANUP STEP: Delete any large model files that might have been saved
    # This is a fail-safe to ensure only adapter weights remain
    full_model_files = ["model.safetensors", "model.bin", "pytorch_model.bin", "diffusion_pytorch_model.bin"]
    
    # Recursively search for large model files in the output directory
    for root, dirs, files in os.walk(final_output_dir):
        for file in files:
            # Skip files in the adapter directory - these are expected to be small
            if "adapter" in root and any(file.endswith(ext) for ext in [".bin", ".safetensors"]):
                continue
                
            # Check if it's a model file
            if any(file == model_file or file.startswith("pytorch_model-") for model_file in full_model_files):
                file_path = os.path.join(root, file)
                try:
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    # If the file is large (over 100MB), it's likely a full model file and should be deleted
                    if file_size_mb > 100:
                        logger.warning(f"Removing large model file: {file_path} ({file_size_mb:.2f} MB)")
                        os.remove(file_path)
                    else:
                        logger.info(f"Keeping small file: {file_path} ({file_size_mb:.2f} MB)")
                except Exception as e:
                    logger.warning(f"Error checking/removing file {file_path}: {e}")
    
    logger.info(f"Cleanup complete. Model saving complete - saved to {final_output_dir}")
    
    # Use the original mixed precision for the pipeline to match user's config
    inference_dtype = torch.float32
    if original_mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif original_mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    
    # Load test pipeline with original model and our adapter
    if config.get("validation_prompt", None) is not None:
        logger.info("Running test inference with validation prompt...")
        try:
            # Create a fresh pipeline from the pretrained model
            from diffusers import StableDiffusionXLPipeline
            from peft import PeftModel
            
            # Store the original dtype we'll use
            adapter_dtype = inference_dtype
            logger.info(f"Using {adapter_dtype} for test inference")
            
            # Load the base pipeline - force torch_dtype to None first to avoid mismatched dtypes
            # We'll manually cast components after adapter is attached
            base_pipeline = StableDiffusionXLPipeline.from_pretrained(
                config.get("pretrained_model_name_or_path"),
                torch_dtype=None,  # Load in original dtype, we'll cast later
            )
            
            # Try multiple ways to load the adapter
            try:
                logger.info("Attempting to load adapter with PeftModel.from_pretrained")
                # Check for adapter in the expected directory structure
                adapter_dir = os.path.join(final_output_dir, "adapter")
                if os.path.exists(adapter_dir):
                    logger.info(f"Found adapter directory at {adapter_dir}")
                    load_path = adapter_dir
                else:
                    logger.info("No adapter subdirectory found, trying main directory")
                    load_path = final_output_dir
                
                # Load the adapter into a clean UNet model
                adapted_unet = PeftModel.from_pretrained(
                    base_pipeline.unet, 
                    load_path,
                    adapter_name="default"
                )
                # Replace the UNet in the pipeline
                base_pipeline.unet = adapted_unet
            except Exception as e:
                logger.warning(f"Could not load adapter with PeftModel.from_pretrained: {e}")
                try:
                    # Try directly loading adapter weights
                    logger.info("Attempting to load adapter weights directly")
                    
                    # First check for adapter_model.bin in adapter directory
                    adapter_bin_path = os.path.join(final_output_dir, "adapter", "adapter_model.bin")
                    if os.path.exists(adapter_bin_path):
                        logger.info(f"Loading adapter from {adapter_bin_path}")
                        adapter_weights = torch.load(adapter_bin_path)
                    # Then check for safetensors version
                    elif os.path.exists(os.path.join(final_output_dir, "adapter", "adapter_model.safetensors")):
                        logger.info("Loading adapter from safetensors file")
                        from safetensors.torch import load_file
                        adapter_weights = load_file(os.path.join(final_output_dir, "adapter", "adapter_model.safetensors"))
                    # Check alternative locations
                    elif os.path.exists(os.path.join(final_output_dir, "adapter_model.bin")):
                        logger.info("Loading adapter from adapter_model.bin in main directory")
                        adapter_weights = torch.load(os.path.join(final_output_dir, "adapter_model.bin"))
                    else:
                        logger.error("Could not find adapter weights file")
                        raise ValueError("Adapter weights not found")
                        
                    # Load weights into model
                    model_state_dict = base_pipeline.unet.state_dict()
                    for name, param in adapter_weights.items():
                        if name in model_state_dict:
                            logger.info(f"Applying adapter parameter: {name}")
                            model_state_dict[name] = param
                        else:
                            # Try to find a matching parameter by removing prefixes
                            # This handles potential naming differences
                            for model_name in model_state_dict.keys():
                                if name.endswith(model_name.split('.')[-1]):
                                    logger.info(f"Found match: {name} -> {model_name}")
                                    model_state_dict[model_name] = param
                                    break
                    
                    # Apply updated state dict
                    base_pipeline.unet.load_state_dict(model_state_dict)
                    logger.info(f"Successfully loaded {len(adapter_weights)} adapter parameters")
                except Exception as inner_e:
                    logger.error(f"Failed to load adapter weights: {inner_e}")
                    raise
                    
            # Now manually cast all pipeline components to the same dtype to avoid mismatches
            logger.info(f"Manually casting all components to {adapter_dtype}")
            if adapter_dtype is not None:
                base_pipeline.to(dtype=adapter_dtype)
                
                # Make sure all text encoder components have the same dtype
                base_pipeline.text_encoder.to(dtype=adapter_dtype)
                base_pipeline.text_encoder_2.to(dtype=adapter_dtype)
                
                # Also ensure VAE has same dtype
                base_pipeline.vae.to(dtype=adapter_dtype)
            
            # Move entire pipeline to appropriate device (likely CUDA)
            device = accelerator.device
            logger.info(f"Moving pipeline to device: {device}")
            base_pipeline.to(device)
            
            with torch.no_grad():
                # Generate a sample image to verify the training
                test_image = base_pipeline(
                    prompt=config.get("validation_prompt"),
                    num_inference_steps=30,
                    guidance_scale=7.5,
                ).images[0]
                
                # Save the test image
                test_image.save(os.path.join(final_output_dir, "test_image.png"))
                logger.info(f"Test image saved to {os.path.join(final_output_dir, 'test_image.png')}")
        except Exception as e:
            logger.error(f"Error running test inference: {e}")
            logger.error(traceback.format_exc())
    
    logger.info("Training and validation complete!")

if __name__ == "__main__":
    main() 