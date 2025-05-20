import os
import sys
import torch
import logging
import numpy as np
import yaml
import json
import re # Added for regex parsing
import ast # Added for safely evaluating list literal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from contextlib import nullcontext
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import is_wandb_available
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from safetensors.torch import load_file
import traceback # Added for better error logging

# Add local peft to the path
sys.path.insert(0, os.path.abspath('./'))

# Import from local peft
from peft import PeftModel, get_peft_model
from peft.tuners.lora import LoraConfig
from peft.tuners.loha import LoHaConfig
from peft.tuners.lokr import LoKrConfig
from peft.tuners.vera import VeraConfig
from peft.tuners.fourierft import FourierFTConfig
from peft.tuners.adalora import AdaLoraConfig
from peft.tuners.waveft import WaveFTConfig
from peft import set_peft_model_state_dict, get_peft_model_state_dict

from datetime import datetime


if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)


def detect_adapter_config(adapter_path: str) -> Dict:
    """
    Detect adapter configuration from saved files.
    
    Args:
        adapter_path: Path to the adapter weights or directory
        
    Returns:
        Dictionary with adapter configuration
    """
    # Default configuration
    config = {
        "adapter_type": "lora",  # Default adapter type
        "rank": 4,               # Default rank
        "alpha": 4,              # Default alpha
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]  # Default target modules
    }
    
    # Handle None adapter_path (for baseline)
    if adapter_path is None:
        logger.info("No adapter path provided (baseline), using default configuration")
        return config
    
    logger.info(f"Detecting adapter configuration from {adapter_path}")
    
    # Check if adapter_path is a directory and contains adapter_config.json
    adapter_dir = adapter_path
    if os.path.isfile(adapter_path):
        adapter_dir = os.path.dirname(adapter_path)
    
    # Look for config in specific locations
    possible_config_paths = [
        os.path.join(adapter_dir, "adapter_config.json"),          # In adapter directory
        os.path.join(adapter_dir, "..", "adapter_config.json"),    # In parent directory
        os.path.join(adapter_dir, "..", "evaluation", "adapter_config.json"),  # In evaluation subdirectory
        os.path.join(adapter_dir, "training_config.yaml"),         # Training config in YAML format
        os.path.join(adapter_dir, "..", "training_config.yaml"),   # Training config in parent directory
    ]
    
    for config_path in possible_config_paths:
        if os.path.exists(config_path):
            logger.info(f"Found configuration at {config_path}")
            
            # Load config based on file extension
            if config_path.endswith(".json"):
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                
                # Extract adapter type from config
                if "peft_type" in file_config:
                    # Map PEFT types to adapter types
                    peft_to_adapter = {
                        "LORA": "lora",
                        "LOHA": "loha",
                        "LOKR": "lokr",
                        "VERA": "vera",
                        "DORA": "dora",
                        "FOURIERFT": "fourierft",
                        "ADALORA": "adalora",
                        "WAVEFT": "waveft"
                    }
                    config["adapter_type"] = peft_to_adapter.get(file_config["peft_type"], "lora")
                
                # Extract rank and alpha
                if "r" in file_config:
                    config["rank"] = file_config["r"]
                    # If only r is specified, default alpha to r
                    config["alpha"] = file_config.get("lora_alpha", file_config.get("alpha", file_config["r"]))
                
                # Extract n_frequency for Fourier/Wave adapters
                if "n_frequency" in file_config:
                    config["n_frequency"] = file_config["n_frequency"]
                
                # Extract scaling for Fourier/Wave adapters
                if "scaling" in file_config:
                    config["scaling"] = file_config["scaling"]
                    
                # Extract wavelet parameters for WaveFT
                if "wavelet_family" in file_config:
                    config["wavelet_family"] = file_config["wavelet_family"]
                
                if "use_idwt" in file_config:
                    config["use_idwt"] = file_config["use_idwt"]
                    
                if "random_loc_seed" in file_config:
                    config["random_loc_seed"] = file_config["random_loc_seed"]
                
                # Extract target modules
                if "target_modules" in file_config:
                    config["target_modules"] = file_config["target_modules"]
            
            elif config_path.endswith(".yaml"):
                with open(config_path, "r") as f:
                    file_config = yaml.safe_load(f)
                
                # Extract adapter configuration from YAML
                if "adapter_type" in file_config:
                    config["adapter_type"] = file_config["adapter_type"]
                
                if "rank" in file_config:
                    config["rank"] = file_config["rank"]
                
                if "alpha" in file_config:
                    config["alpha"] = file_config["alpha"]
                
                # Extract n_frequency for Fourier/Wave adapters
                if "n_frequency" in file_config:
                    config["n_frequency"] = file_config["n_frequency"]
                
                # Extract scaling for Fourier/Wave adapters
                if "scaling" in file_config:
                    config["scaling"] = file_config["scaling"]
                    
                # Extract wavelet parameters for WaveFT
                if "wavelet_family" in file_config:
                    config["wavelet_family"] = file_config["wavelet_family"]
                
                if "use_idwt" in file_config:
                    config["use_idwt"] = file_config["use_idwt"]
                    
                if "random_loc_seed" in file_config:
                    config["random_loc_seed"] = file_config["random_loc_seed"]
                
                if "target_modules" in file_config:
                    config["target_modules"] = file_config["target_modules"]
            
            # Successfully loaded config
            logger.info(f"Detected adapter configuration: {config}")
            return config
    
    # If no configuration file found, try to infer from file structure and naming
    if os.path.isfile(adapter_path):
        filename = os.path.basename(adapter_path)
        if "loha" in filename.lower():
            config["adapter_type"] = "loha"
        elif "lokr" in filename.lower():
            config["adapter_type"] = "lokr"
        elif "vera" in filename.lower():
            config["adapter_type"] = "vera"
        elif "dora" in filename.lower():
            config["adapter_type"] = "dora"
        elif "fourier" in filename.lower():
            config["adapter_type"] = "fourierft"
        elif "adalora" in filename.lower():
            config["adapter_type"] = "adalora"
        elif "wave" in filename.lower():
            config["adapter_type"] = "waveft"
    
    logger.info(f"No configuration file found, using default configuration: {config}")
    return config


def load_inference_pipeline(
    pretrained_model_path: str,
    adapter_path: str,
    adapter_name: str = None,
    adapter_config: Optional[Dict] = None,
    torch_dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    use_xformers: bool = True
) -> StableDiffusionXLPipeline:
    """
    Load a StableDiffusionXLPipeline with a trained adapter using the approach from inference.py.
    
    Args:
        pretrained_model_path: Path to the pretrained SDXL model
        adapter_path: Path to the trained adapter weights, or None for baseline
        adapter_name: Name of the adapter method (for logging)
        adapter_config: Optional configuration for the adapter
        torch_dtype: Data type for the model
        device: Device to load the model on
        use_xformers: Whether to use xformers for memory-efficient attention
        
    Returns:
        The loaded pipeline with the adapter applied
    """
    # If adapter_path is None, this is the baseline method (no adapter)
    if adapter_path is None:
        logger.info(f"Loading baseline pipeline from {pretrained_model_path} without any adapter")
        
        # Load the base pipeline without any adapter
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=None,  # Important: Don't convert dtype yet
            safety_checker=None,
        )
        
        # Move the pipeline to the device and set dtype after loading
        pipeline.to(device=device, dtype=torch_dtype)
        
        # Set the scheduler (ensures consistent results)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
        )
        
        # Enable xformers if available
        if use_xformers and torch.cuda.is_available():
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Using xformers memory efficient attention")
            except ImportError:
                logger.warning("xformers is not available, continuing without it")
        
        # Set progress bar config
        pipeline.set_progress_bar_config(disable=True)
        
        return pipeline
    
    logger.info(f"Loading inference pipeline from {pretrained_model_path} with adapter from {adapter_path}")
    
    # Important: Load the base pipeline WITHOUT dtype for now
    # We'll manually cast everything after loading the adapter
    logger.info(f"Loading base pipeline from {pretrained_model_path}")
    base_pipeline = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_path,
        torch_dtype=None,  # Important: Don't convert dtype yet
        safety_checker=None,
    )

    # Flag to track adapter loading success
    adapter_loaded = False
    from peft import PeftModel # Ensure PeftModel is imported

    # Determine the base adapter directory
    adapter_dir = os.path.dirname(adapter_path) if os.path.isfile(adapter_path) else adapter_path

    # --- Try loading adapter using PeftModel ---
    # Attempt 1: Load from the main directory
    try:
        logger.info(f"Attempt 1: Loading adapter with PeftModel from main directory: {adapter_dir}")
        adapted_unet = PeftModel.from_pretrained(
            base_pipeline.unet,
            adapter_dir,
            adapter_name="default"
        )
        base_pipeline.unet = adapted_unet
        adapter_loaded = True
        logger.info("Successfully loaded adapter from main directory.")
    except Exception as e1:
        logger.warning(f"Failed loading from main directory {adapter_dir}: {e1}")
        # Optionally log more details for debugging: logger.debug(traceback.format_exc())

    # Attempt 2: Load from the 'adapter' subdirectory if Attempt 1 failed
    if not adapter_loaded:
        adapter_subdir = os.path.join(adapter_dir, "adapter")
        if os.path.exists(adapter_subdir):
            try:
                logger.info(f"Attempt 2: Loading adapter with PeftModel from subdirectory: {adapter_subdir}")
                adapted_unet = PeftModel.from_pretrained(
                    base_pipeline.unet,
                    adapter_subdir,
                    adapter_name="default"
                )
                base_pipeline.unet = adapted_unet
                adapter_loaded = True
                logger.info("Successfully loaded adapter from adapter subdirectory.")
            except Exception as e2:
                logger.warning(f"Failed loading from subdirectory {adapter_subdir}: {e2}")
                # Optionally log more details for debugging: logger.debug(traceback.format_exc())
        else:
            logger.info(f"Adapter subdirectory {adapter_subdir} does not exist. Skipping.")

    # --- Fallback: Try direct weight loading if PeftModel failed ---
    if not adapter_loaded:
        logger.warning("Could not load adapter using PeftModel. Trying direct weight loading...")
        try:
            logger.info(f"Attempting to load adapter weights directly from {adapter_path}")

            # Check for safetensors file
            if adapter_path.endswith('.safetensors'):
                logger.info(f"Loading adapter from safetensors file: {adapter_path}")
                from safetensors.torch import load_file
                adapter_weights = load_file(adapter_path, map_location="cpu") # Load to CPU first
            # Check for .bin or other pytorch formats
            elif adapter_path.endswith(('.bin', '.pt', '.pth', '.ckpt')):
                 logger.info(f"Loading adapter from PyTorch file: {adapter_path}")
                 adapter_weights = torch.load(adapter_path, map_location="cpu") # Load to CPU first
            else:
                logger.warning(f"Unrecognized adapter file extension for direct loading: {adapter_path}. Skipping direct load.")
                adapter_weights = None # Ensure adapter_weights is defined

            if adapter_weights:
                # Load weights into model state dict
                model_state_dict = base_pipeline.unet.state_dict()
                loaded_params = 0
                skipped_params = 0

                # Handle potential prefixes in adapter keys (common in PEFT)
                possible_prefixes = ["base_model.model.", ""] # Add more if needed
                new_state_dict = model_state_dict.copy()

                for adapter_key, param in adapter_weights.items():
                    found_match = False
                    for prefix in possible_prefixes:
                        model_key = prefix + adapter_key # Try direct match first
                        if model_key in new_state_dict:
                             if new_state_dict[model_key].shape == param.shape:
                                new_state_dict[model_key] = param
                                loaded_params += 1
                                found_match = True
                                break
                             else:
                                logger.warning(f"Shape mismatch for key {model_key}: model is {new_state_dict[model_key].shape}, adapter is {param.shape}. Skipping.")
                                skipped_params += 1
                                found_match = True # Mark as found to avoid suffix check
                                break

                    # If direct match failed, try suffix matching as a fallback
                    if not found_match:
                        matched_by_suffix = False
                        for model_key_suffix_check in new_state_dict.keys():
                           # Check if the adapter key ends with the model key (or vice versa, less common)
                           if adapter_key.endswith(model_key_suffix_check) or model_key_suffix_check.endswith(adapter_key):
                               if new_state_dict[model_key_suffix_check].shape == param.shape:
                                   logger.info(f"Suffix match: {adapter_key} -> {model_key_suffix_check}")
                                   new_state_dict[model_key_suffix_check] = param
                                   loaded_params += 1
                                   matched_by_suffix = True
                                   break
                               else:
                                   logger.warning(f"Suffix match found, but shape mismatch for key {model_key_suffix_check}: model is {new_state_dict[model_key_suffix_check].shape}, adapter is {param.shape}. Skipping.")
                                   skipped_params += 1
                                   matched_by_suffix = True # Mark as found to avoid warning
                                   break
                        if not matched_by_suffix:
                            logger.warning(f"Could not find matching key in model state_dict for adapter key: {adapter_key}. Skipping.")
                            skipped_params += 1


                # Apply updated state dict if any params were loaded
                if loaded_params > 0:
                     incompatible_keys = base_pipeline.unet.load_state_dict(new_state_dict, strict=False)
                     if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                         logger.warning(f"Incompatible keys found when loading state dict directly:")
                         if incompatible_keys.missing_keys: logger.warning(f" Missing keys: {incompatible_keys.missing_keys}")
                         if incompatible_keys.unexpected_keys: logger.warning(f" Unexpected keys: {incompatible_keys.unexpected_keys}")

                     logger.info(f"Successfully loaded {loaded_params}/{loaded_params + skipped_params} adapter parameters via direct loading.")
                     adapter_loaded = True
                else:
                    logger.warning("Direct weight loading failed: No matching parameters found or applied.")

        except Exception as e:
            logger.error(f"Failed to load adapter weights directly: {e}")
            import traceback
            logger.error(traceback.format_exc()) # Log full traceback for debugging

    # Final check if adapter was loaded by any method
    if not adapter_loaded:
        logger.error("Failed to load adapter using any method. Proceeding with the base model only.")
        # Optionally raise an error here if adapter loading is critical
        # raise RuntimeError("Adapter loading failed.")

    # Now manually cast all pipeline components to the same dtype
    logger.info(f"Manually casting all components to {torch_dtype}")
    
    # Move entire pipeline to GPU device and set dtype
    logger.info(f"Moving pipeline to device: {device}")
    base_pipeline.to(device=device, dtype=torch_dtype)
    
    # Set the scheduler (ensures consistent results)
    base_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        base_pipeline.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2,
    )
    
    # Enable xformers if available
    if use_xformers and torch.cuda.is_available():
        try:
            base_pipeline.unet.enable_xformers_memory_efficient_attention()
            logger.info("Using xformers for memory-efficient attention")
        except Exception as e:
            logger.warning(f"Failed to enable xformers: {e}")
    
    # Set progress bar config
    base_pipeline.set_progress_bar_config(disable=True)
    
    return base_pipeline


def render_prompt_templates(
    base_prompts: Dict[str, str],
    unique_token: str = "zjw",
    class_name: str = None,
) -> Dict[str, str]:
    """
    Render prompt templates for evaluation.
    
    Args:
        base_prompts: Dictionary of prompt templates keyed by prompt type
        unique_token: The unique token used during training (e.g., "zjw")
        class_name: The class name for the instance
        
    Returns:
        Dictionary of rendered prompts
    """
    # Check if base_prompts is empty or None
    if not base_prompts:
        logger.error("Cannot render prompts: base_prompts dictionary is empty or None.")
        return {} # Return empty dict if no base prompts provided

    rendered_prompts = {}

    for prompt_type, prompt_template in base_prompts.items():
        # Always replace the placeholders, regardless of the prompt type
        rendered_prompt = prompt_template

        # Replace [unique_token] with the actual unique token
        if "[unique_token]" in rendered_prompt:
            rendered_prompt = rendered_prompt.replace("[unique_token]", unique_token)

        # Replace [class_name] with the actual class name if provided
        if class_name and "[class_name]" in rendered_prompt:
            rendered_prompt = rendered_prompt.replace("[class_name]", class_name)

        rendered_prompts[prompt_type] = rendered_prompt

    return rendered_prompts


def generate_samples(
    pipeline: StableDiffusionXLPipeline,
    prompts: Union[str, List[str]],
    output_dir: str,
    num_images_per_prompt: int = 4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
    seed: Optional[int] = None,
    height: int = 1024,
    width: int = 1024,
    autocast_dtype: Optional[torch.dtype] = None,
    add_watermark: bool = False,
    unique_seeds: Optional[List[int]] = None,
) -> List[Image.Image]:
    """
    Generate samples using the provided pipeline and prompts.
    
    Args:
        pipeline: The StableDiffusionXLPipeline to use
        prompts: The prompt or list of prompts to generate images from
        output_dir: Directory to save the generated images
        num_images_per_prompt: Number of images to generate per prompt
        guidance_scale: The classifier-free guidance scale
        num_inference_steps: Number of denoising steps
        seed: Random seed for reproducibility
        height: Height of the generated images
        width: Width of the generated images
        autocast_dtype: Data type to use for autocast
        add_watermark: Whether to add a watermark to the generated images
        unique_seeds: List of unique seeds for each image
        
    Returns:
        List of generated images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(prompts, str):
        prompts = [prompts]
    
    all_images = []
    
    # Strong negative prompt for better quality
    negative_prompt = "low quality, bad anatomy, worst quality, low res, blurry, distorted, deformed, ugly, duplicate, morbid, mutilated, mutation, disfigured, poorly drawn face, bad proportions"
    
    # Create a context for inference
    if autocast_dtype is None:
        # Just use nullcontext if no specific dtype is provided
        inference_context = nullcontext()
    else:
        inference_context = torch.autocast(pipeline.device.type, dtype=autocast_dtype)
    
    # Initialize random if needed and no seed is provided
    import random
    if seed is None and unique_seeds is None:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using random seed: {seed}")
    
    # Generate seeds for each image if not provided
    if unique_seeds is None:
        # Use base seed to generate unique seeds for each image
        base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        unique_seeds = [base_seed + i * 1000 for i in range(num_images_per_prompt)]
        logger.info(f"Generated seeds for images: {unique_seeds}")
    
    # Ensure we have enough seeds
    if len(unique_seeds) < num_images_per_prompt:
        # Extend the list if needed
        additional_seeds = [unique_seeds[-1] + i * 1000 for i in range(1, num_images_per_prompt - len(unique_seeds) + 1)]
        unique_seeds.extend(additional_seeds)
        logger.info(f"Extended seeds list to: {unique_seeds}")
    
    # Limit to requested number
    unique_seeds = unique_seeds[:num_images_per_prompt]
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        logger.info(f"Generating images for prompt: {prompt}")
        
        for j in range(num_images_per_prompt):
            try:
                # Use the seed for this specific image
                current_seed = unique_seeds[j]
                logger.info(f"Using seed {current_seed} for image {j+1}")
                generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
                
                # Generate the image with safeguards
                with inference_context:
                    output = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        output_type="pil",  # Ensure we get PIL images
                    )
                
                # Get the image from the output
                image = output.images[0]
                
                # Validate the image
                if image is None:
                    logger.warning(f"Pipeline returned None for image. Using placeholder.")
                    # Create a placeholder gradient image
                    image = Image.new('RGB', (width, height))
                    # Create a simple gradient as placeholder
                    for y in range(height):
                        for x in range(width):
                            r = int(255 * x / width)
                            g = int(255 * y / height)
                            b = 120
                            image.putpixel((x, y), (r, g, b))
                
                # Add the image to our collection
                all_images.append(image)
                
                # Save the image with a descriptive filename including seed
                prompt_hash = abs(hash(prompt)) % 10000
                image_filename = f"prompt_{prompt_hash}_sample_{j}_seed_{current_seed}.png"
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)
                logger.info(f"Saved image to {image_path}")
                
                # Save the prompt information for reference
                prompt_info_path = os.path.join(output_dir, f"prompt_{prompt_hash}_info.txt")
                with open(prompt_info_path, "w") as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Negative Prompt: {negative_prompt}\n")
                    f.write(f"Guidance Scale: {guidance_scale}\n")
                    f.write(f"Inference Steps: {num_inference_steps}\n")
                    f.write(f"Seed: {current_seed}\n")
                
                # Clean up GPU memory after each generation
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error generating image: {str(e)}")
                # Create a placeholder gradient image for errors
                placeholder = Image.new('RGB', (width, height))
                # Use a distinct color gradient for error cases
                for y in range(height):
                    for x in range(width):
                        r = int(180 + 75 * x / width)
                        g = int(100 + 50 * y / height)
                        b = int(50 + 150 * x / width)
                        placeholder.putpixel((x, y), (r, g, b))
                
                all_images.append(placeholder)
                
                # Save the placeholder
                prompt_hash = abs(hash(prompt)) % 10000
                image_filename = f"prompt_{prompt_hash}_sample_{j}_error_placeholder.png"
                image_path = os.path.join(output_dir, image_filename)
                placeholder.save(image_path)
                logger.warning(f"Saved placeholder image to {image_path}")
                
                # Clean up GPU memory
                torch.cuda.empty_cache()
    
    return all_images


def log_evaluation_results(
    images: List[Image.Image],
    prompts: List[str],
    method_name: str,
    concept_name: str,
    tracker: Optional[object] = None,
    epoch: Optional[int] = None,
    is_final_validation: bool = False
):
    """
    Log evaluation results to the tracker (tensorboard, wandb, etc.)
    
    Args:
        images: List of generated images
        prompts: Corresponding prompts for the images
        method_name: Name of the adapter method
        concept_name: Name of the concept being generated
        tracker: Tracking object (tensorboard, wandb, etc.)
        epoch: Current training epoch, if applicable
        is_final_validation: Whether this is the final validation
    """
    if tracker is None:
        return
    
    phase_name = "test" if is_final_validation else "validation"
    
    if hasattr(tracker, "name") and tracker.name == "tensorboard":
        np_images = np.stack([np.asarray(img) for img in images])
        tracker.writer.add_images(
            f"{phase_name}/{method_name}/{concept_name}",
            np_images,
            epoch if epoch is not None else 0,
            dataformats="NHWC"
        )
    
    if is_wandb_available() and hasattr(tracker, "name") and tracker.name == "wandb":
        wandb_images = [
            wandb.Image(image, caption=f"{method_name}-{concept_name}-{i}: {prompts[i % len(prompts)]}")
            for i, image in enumerate(images)
        ]
        tracker.log({f"{phase_name}/{method_name}/{concept_name}": wandb_images})


def batch_evaluation(
    config: Dict,
    adapter_paths: Dict[str, str],
    concept_names: List[str],
    prompts_file: str,
    output_base_dir: str,
    pretrained_model_path: str,
    device: str = "cuda",
    num_images_per_prompt: int = 4,
    torch_dtype: torch.dtype = torch.float16,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
    seed: Optional[int] = None,
    tracker: Optional[object] = None,
    save_grid: bool = False,
    evaluation_params: Optional[Dict] = None,
):
    """
    Run batch evaluation of multiple adapter methods on multiple concepts.
    
    Args:
        config: Configuration for the evaluation
        adapter_paths: Dictionary mapping method names to adapter paths
        concept_names: List of concept names to evaluate
        prompts_file: Path to the prompts file
        output_base_dir: Base directory for saving outputs
        pretrained_model_path: Path to the pretrained SDXL model
        device: Device to run evaluation on
        num_images_per_prompt: Number of images to generate per prompt
        torch_dtype: Data type for the model
        guidance_scale: The classifier-free guidance scale
        num_inference_steps: Number of denoising steps
        seed: Random seed for reproducibility
        tracker: Tracking object (tensorboard, wandb, etc.)
        save_grid: Whether to save a grid of all generated images
        evaluation_params: Additional evaluation parameters
    """
    # Default evaluation params
    if evaluation_params is None:
        evaluation_params = {}
    
    max_prompts = evaluation_params.get("max_prompts", None)
    use_unique_seeds = evaluation_params.get("use_unique_seeds", False)
    
    # Override default parameters with those from evaluation_params if provided
    guidance_scale = evaluation_params.get("guidance_scale", guidance_scale)
    num_inference_steps = evaluation_params.get("num_inference_steps", num_inference_steps)
    
    logger.info(f"Starting batch evaluation of {len(adapter_paths)} methods on {len(concept_names)} concepts")
    logger.info(f"Evaluation parameters: max_prompts={max_prompts}, use_unique_seeds={use_unique_seeds}")
    logger.info(f"Generation parameters: guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps}")
    
    # Import random module for prompt selection
    import random
    # Set random seed for reproducibility in prompt selection
    random.seed(seed)
    
    # Load prompts and class map from file
    # This now returns both templates and the class mapping
    prompt_templates, concept_to_class_map = load_prompts_from_file(prompts_file)
    
    logger.info(f"Loaded {len(prompt_templates)} prompt templates.")
    logger.info(f"Loaded {len(concept_to_class_map)} concept-to-class mappings.")
    
    # Randomly select prompts if max_prompts is specified
    if max_prompts is not None and max_prompts > 0 and max_prompts < len(prompt_templates):
        prompt_keys = list(prompt_templates.keys())
        selected_keys = random.sample(prompt_keys, max_prompts)
        
        # Create a new dictionary with only the selected prompts
        selected_templates = {k: prompt_templates[k] for k in selected_keys}
        prompt_templates = selected_templates
        
        logger.info(f"Randomly selected {max_prompts} prompts from {len(prompt_keys)} available prompts")
    elif max_prompts is not None and max_prompts == 0:
        logger.info("max_prompts set to 0, skipping image generation.")
        return # Exit if no prompts are selected

    # Dictionary to store all generated images for grid creation
    if save_grid:
        all_method_images = {}
    
    for method_name, adapter_path in adapter_paths.items():
        logger.info(f"Evaluating method: {method_name}")
        
        # Detect adapter configuration
        adapter_config = None
        if adapter_path is not None:
            adapter_config = detect_adapter_config(adapter_path)
            logger.info(f"Using {adapter_config['adapter_type']} adapter with rank={adapter_config['rank']}")
        else:
            logger.info("Using baseline model without adapter")
        
        # Load the pipeline with the current adapter
        pipeline = load_inference_pipeline(
            pretrained_model_path=pretrained_model_path,
            adapter_path=adapter_path,
            adapter_name=method_name,
            adapter_config=adapter_config,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        if save_grid:
            all_method_images[method_name] = {}
        
        for concept_name in concept_names:
            logger.info(f"Generating samples for concept: {concept_name}")
            
            # Get the class name for the concept using the loaded map
            class_name = concept_to_class_map.get(concept_name)
            if class_name is None:
                # Fallback: Use the concept name itself (or process it simply)
                class_name = concept_name.replace("_", " ")
                logger.warning(f"Concept '{concept_name}' not found in class map from {prompts_file}. Using fallback class name: '{class_name}'")
                # Optionally, use the config-based lookup as another fallback
                # class_name = get_class_name_from_concept(concept_name, config)
                # logger.warning(f"Using class name from config/fallback logic: '{class_name}'")

            logger.info(f"Using class name '{class_name}' for concept '{concept_name}'")
            
            # Render prompts for the current concept
            rendered_prompts = render_prompt_templates(
                prompt_templates,
                unique_token=config.get("unique_token", "zjw"),
                class_name=class_name, # Use the correctly determined class name
            )
            
            # Create output directory for this method and concept
            output_dir = os.path.join(output_base_dir, method_name, concept_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate samples for each prompt type
            all_images = []
            all_prompts = []
            
            # Store images by prompt type for the grid
            if save_grid:
                all_method_images[method_name][concept_name] = {}
            
            for prompt_type, prompt in rendered_prompts.items():
                prompt_output_dir = os.path.join(output_dir, prompt_type)
                
                # Generate seeds for each image if using unique seeds
                image_seeds = None
                if use_unique_seeds:
                    base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
                    image_seeds = [base_seed + i for i in range(num_images_per_prompt)]
                    # logger.info(f"Using unique seeds for each image: {image_seeds}") # Less verbose logging
                
                images = generate_samples(
                    pipeline=pipeline,
                    prompts=prompt,
                    output_dir=prompt_output_dir,
                    num_images_per_prompt=num_images_per_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    autocast_dtype=torch_dtype,
                    unique_seeds=image_seeds,
                )
                
                all_images.extend(images)
                all_prompts.extend([prompt] * len(images))
                
                # Store images for grid creation
                if save_grid:
                    all_method_images[method_name][concept_name][prompt_type] = images
            
            # Log evaluation results
            log_evaluation_results(
                images=all_images,
                prompts=all_prompts,
                method_name=method_name,
                concept_name=concept_name,
                tracker=tracker,
                is_final_validation=True,
            )
        
        # Clean up to conserve GPU memory
        del pipeline
        torch.cuda.empty_cache()
    
    # Create and save the grid if requested
    if save_grid:
        logger.info("Creating image grids for comparison")
        for concept_name in concept_names:
            # Determine the class name again for the grid function's prompt rendering
            class_name_for_grid = concept_to_class_map.get(concept_name, concept_name.replace("_", " "))
            
            # Prepare rendered prompts for the grid titles
            grid_rendered_prompts = render_prompt_templates(
                prompt_templates,
                unique_token=config.get("unique_token", "zjw"),
                class_name=class_name_for_grid,
            )

            for prompt_type in prompt_templates.keys():
                # Check if we have images for this prompt type for all methods
                methods_with_images = []
                for method_name in adapter_paths.keys():
                    if (method_name in all_method_images and 
                        concept_name in all_method_images[method_name] and 
                        prompt_type in all_method_images[method_name][concept_name] and
                        len(all_method_images[method_name][concept_name][prompt_type]) > 0): # Ensure list is not empty
                        methods_with_images.append(method_name)
                
                if not methods_with_images:
                    logger.debug(f"Skipping grid for {concept_name}, {prompt_type}: No methods have images.")
                    continue

                # Check if the prompt_type exists in the rendered prompts for the grid
                if prompt_type not in grid_rendered_prompts:
                    logger.warning(f"Prompt type '{prompt_type}' not found in rendered prompts for grid title generation. Skipping grid.")
                    continue

                # Create a grid for this prompt type and concept
                logger.info(f"Creating grid for {concept_name} with prompt type {prompt_type}")
                create_comparison_grid(
                    all_method_images=all_method_images,
                    concept_name=concept_name,
                    prompt_type=prompt_type,
                    methods=methods_with_images,
                    output_base_dir=output_base_dir,
                    rendered_prompts=grid_rendered_prompts, # Pass the correctly rendered prompts
                )


def create_comparison_grid(
    all_method_images: Dict[str, Dict[str, Dict[str, List[Image.Image]]]],
    concept_name: str,
    prompt_type: str,
    methods: List[str],
    output_base_dir: str,
    rendered_prompts: Dict[str, str],
):
    """
    Create a grid comparing images from different methods for the same concept and prompt type.
    
    Args:
        all_method_images: Dictionary of all generated images
        concept_name: Name of the concept
        prompt_type: Type of the prompt
        methods: List of method names to include in the grid
        output_base_dir: Base directory for saving outputs
        rendered_prompts: Dictionary of rendered prompt templates
    """
    # Define grid dimensions
    num_methods = len(methods)
    num_samples = min(len(all_method_images[method][concept_name][prompt_type]) for method in methods)
    
    if num_samples == 0:
        logger.warning(f"No images found for grid: {concept_name}, {prompt_type}")
        return
    
    # Get the first image to determine dimensions
    first_image = all_method_images[methods[0]][concept_name][prompt_type][0]
    image_width, image_height = first_image.size
    
    # Add space for headers and method names
    header_height = 40
    method_label_height = 30
    grid_width = image_width * num_samples
    grid_height = (image_height + method_label_height) * num_methods + header_height
    
    # Create a blank grid image
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_image)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Add the prompt text as a header
    prompt = rendered_prompts[prompt_type]
    draw.text((10, 10), f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}", fill=(0, 0, 0), font=font)
    
    # Place images in the grid
    for m_idx, method in enumerate(methods):
        # Add method name
        y_offset = header_height + m_idx * (image_height + method_label_height)
        draw.text((10, y_offset), method, fill=(0, 0, 0), font=font)
        
        # Add images for this method
        for s_idx in range(min(num_samples, len(all_method_images[method][concept_name][prompt_type]))):
            img = all_method_images[method][concept_name][prompt_type][s_idx]
            x_offset = s_idx * image_width
            grid_image.paste(img, (x_offset, y_offset + method_label_height))
    
    # Save the grid
    grid_dir = os.path.join(output_base_dir, "comparison_grids")
    os.makedirs(grid_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_path = os.path.join(grid_dir, f"{concept_name}_{prompt_type}_{timestamp}.png")
    grid_image.save(grid_path)
    logger.info(f"Saved comparison grid to {grid_path}")


def load_prompts_from_file(prompts_file: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load prompt templates and class mappings from a prompts file.
    Assumes the format has a 'Classes' section and a 'Prompts' section.
    If the file cannot be read or parsed, returns empty dictionaries.

    Args:
        prompts_file: Path to the prompts file

    Returns:
        Tuple containing:
            - Dictionary of prompt templates (key: prompt_type, value: template string) or {} on failure.
            - Dictionary of concept to class mappings (key: concept_name, value: class_name) or {} on failure.
    """
    prompt_templates = {}
    concept_to_class_map = {}

    # No default templates/map defined here anymore.

    if not prompts_file or not os.path.exists(prompts_file):
        logger.error(f"Prompts file not found or path is invalid: {prompts_file}. Cannot load prompts or class map.")
        return {}, {} # Return empty dicts explicitly

    try:
        logger.info(f"Loading prompts and class map from file: {prompts_file}")
        with open(prompts_file, "r") as f:
            content = f.read()

        # --- Parse Classes Section ---
        in_classes_section = False
        parsed_classes = False
        logger.info("Attempting to parse 'Classes' section...")
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Detect section headers robustly (case-insensitive)
            if line.lower() == "classes":
                logger.info(f"Found 'Classes' header at line {line_num}.")
                in_classes_section = True
                continue
            if line.lower() == "prompts":
                logger.info(f"Found 'Prompts' header at line {line_num}, stopping class parsing.")
                in_classes_section = False
                break # Stop processing classes once Prompts section is reached

            if in_classes_section:
                if line.lower() == "subject_name,class": # Skip header
                    continue
                try:
                    # Handle potential extra whitespace around comma
                    parts = [p.strip() for p in line.split(",", 1)]
                    if len(parts) == 2 and parts[0] and parts[1]:
                        concept, class_name = parts
                        concept_to_class_map[concept] = class_name
                        parsed_classes = True
                    else:
                         logger.warning(f"Skipping malformed line (expected 'concept,class') in Classes section at line {line_num}: {line}")
                except ValueError:
                    logger.warning(f"Skipping malformed line (ValueError) in Classes section at line {line_num}: {line}")

        if parsed_classes:
             logger.info(f"Successfully parsed {len(concept_to_class_map)} class mappings.")
        else:
             logger.warning(f"No valid class mappings found or parsed in the 'Classes' section of {prompts_file}.")
             # Keep concept_to_class_map empty, do not use defaults.

        # --- Parse Prompts Section ---
        logger.info("Attempting to parse 'Prompts' section...")
        # Find the start of the Python list assignment for prompts
        prompt_list_match = re.search(r"prompt_list\s*=\s*\[", content, re.IGNORECASE | re.MULTILINE)
        parsed_prompts = False

        if prompt_list_match:
            logger.info("Found 'prompt_list = [' structure. Trying to parse list content...")
            list_start_index = prompt_list_match.end()

            # Find the corresponding closing bracket ']' carefully
            bracket_level = 1
            list_end_index = -1
            in_string = None # Track if inside single or double quotes
            escaped = False
            for i in range(list_start_index, len(content)):
                char = content[i]

                if in_string:
                    if char == in_string and not escaped:
                        in_string = None
                    elif char == '\\' and not escaped:
                        escaped = True
                    else:
                        escaped = False
                else:
                    if char == '[':
                        bracket_level += 1
                    elif char == ']':
                        bracket_level -= 1
                        if bracket_level == 0:
                            list_end_index = i
                            logger.info(f"Found matching ']' at index {i}.")
                            break
                    elif char == '"' or char == "'":
                        in_string = char
                    elif char == '#': # Stop parsing list if comment starts
                        # Check if the # is inside the list or after
                        sub_content = content[list_start_index:i]
                        if sub_content.count('[') == sub_content.count(']'):
                             logger.info(f"Comment detected outside list structure at index {i}, stopping bracket search.")
                             # Treat this as end of list content if brackets are balanced
                             list_end_index = i 
                             break
                        # else: comment is likely inside a string or nested structure, ignore for now

                escaped = False # Reset escape status


            if list_end_index != -1:
                list_content = content[list_start_index:list_end_index].strip()
                logger.info(f"Extracted list content (approx {len(list_content)} chars). Attempting regex extraction of string literals.")
                # Regex to find Python string literals (handles basic escapes)
                # It captures the quote type (' or ") and the content
                string_literal_regex = re.compile(r"(['\"])(.*?)(?<!\\)\1")
                
                found_strings = []
                try:
                    matches = string_literal_regex.finditer(list_content)
                    count = 0
                    for match_num, match in enumerate(matches):
                        # Extract the content within the quotes (group 2)
                        raw_string = match.group(2)
                        
                        # Basic unescaping (you might need a more robust solution for complex cases)
                        # Replace escaped quotes and backslashes
                        unescaped_string = raw_string.replace(f"\\{match.group(1)}", match.group(1)).replace("\\\\", "\\")
                        
                        # Replace placeholders
                        processed = unescaped_string
                        if "{0}" in processed:
                            processed = processed.replace("{0}", "[unique_token]")
                        if "{1}" in processed:
                            processed = processed.replace("{1}", "[class_name]")
                        
                        # Store the processed template
                        key = f"prompt_{count + 1}"
                        prompt_templates[key] = processed.strip()
                        found_strings.append(key)
                        count += 1
                    
                    if count > 0:
                        logger.info(f"Successfully extracted and processed {count} prompt templates using regex from list content.")
                        parsed_prompts = True
                    else:
                        logger.warning("Regex found no string literals within the 'prompt_list = [ ... ]' content.")
                        
                except Exception as regex_err:
                    logger.error(f"Error during regex processing of prompt list content: {regex_err}")
                    logger.debug(f"Content being processed: {list_content[:500]}{'...' if len(list_content) > 500 else ''}")
                    # Ensure we proceed to fallback if regex fails
                    parsed_prompts = False 

                # --- Remove the ast.literal_eval block --- 
                # list_string = f"[{list_content}]"
                # try:
                #     logger.info(f"Attempting ast.literal_eval on: {list_string[:200]}{'...' if len(list_string) > 200 else ''}")
                #     parsed_prompt_list = ast.literal_eval(list_string)
                #     # ... (rest of the old ast parsing logic removed)
                # except (SyntaxError, ValueError, MemoryError) as e:
                #     logger.warning(f"Could not parse prompt_list content using ast.literal_eval: {type(e).__name__}: {e}. Will attempt fallback parsing.")
                #     logger.debug(f"Problematic list string content (partial): {list_string[:500]}{'...' if len(list_string) > 500 else ''}")
                #     prompt_list_match = None # Force fallback (this assignment is wrong here anyway)
            else:
                 logger.warning("Could not find closing bracket ']' for prompt_list. Will attempt fallback parsing.")
                 # No need to set prompt_list_match = None, just let it proceed to fallback

        # Fallback or alternative format: Parse lines like "key: template" under "Prompts" header
        # This will run if prompt_list parsing failed or wasn't found, but only if parsed_prompts is still False
        if not parsed_prompts:
            logger.info("Attempting fallback parsing for prompts (looking for 'key: template' under 'Prompts' header)...")
            in_prompts_section = False
            found_prompts_fallback = False
            for line_num, line in enumerate(content.splitlines(), 1):
                 line = line.strip()
                 if not line or line.startswith("#"):
                     continue
                 # Detect section headers robustly
                 if line.lower() == "prompts":
                     logger.info(f"Found 'Prompts' header at line {line_num} for fallback parsing.")
                     in_prompts_section = True
                     continue
                 if line.lower() == "classes": # Stop if we hit Classes again
                     logger.info(f"Found 'Classes' header at line {line_num}, stopping fallback prompt parsing.")
                     in_prompts_section = False
                     continue # Don't break, might find classes later

                 if in_prompts_section:
                     # Check for 'key: value' format
                     if ":" in line:
                         try:
                             # Split only on the first colon
                             key, template = line.split(":", 1)
                             key = key.strip()
                             template = template.strip()
                             # Basic check for validity (non-empty key/template)
                             if key and template:
                                 # Optionally check for placeholders if strictly required
                                 # if "[unique_token]" in template and "[class_name]" in template:
                                 prompt_templates[key] = template
                                 found_prompts_fallback = True
                                 # else:
                                 #    logger.warning(f"Skipping prompt line {line_num} (key:value format) due to missing placeholders: {line}")
                             else:
                                logger.warning(f"Skipping prompt line {line_num} (key:value format) with empty key or value: {line}")
                         except ValueError:
                             logger.warning(f"Skipping malformed line {line_num} in Prompts section (expected 'key: value'): {line}")
                     # Could add other checks here if needed (e.g., lines without ':')
                     # else:
                     #    logger.warning(f"Skipping line {line_num} in Prompts section (not 'key: value' format): {line}")


            if found_prompts_fallback:
                 logger.info(f"Parsed {len(prompt_templates)} prompts using fallback 'key:value' format.")
                 parsed_prompts = True # Mark as success
            else:
                 logger.warning(f"No valid prompts found in {prompts_file} using fallback 'key:value' format either.")

        # Final check: If no prompts were parsed by any method
        if not parsed_prompts:
            logger.error(f"Failed to parse any prompts from {prompts_file} using available methods.")
            # Ensure prompt_templates is empty
            prompt_templates = {}


    except Exception as e:
        logger.error(f"Critical error loading or parsing prompts file {prompts_file}: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        logger.error("Returning empty prompts and class map due to unexpected error.")
        # Ensure return values are empty dicts on critical failure
        return {}, {}

    logger.info(f"Finished processing prompts file. Found {len(prompt_templates)} prompt templates and {len(concept_to_class_map)} class mappings.")
    # Return the potentially empty dictionaries. The caller must check if they are empty.
    return prompt_templates, concept_to_class_map


def get_class_name_from_concept(concept_name: str, config: Dict) -> str:
    """
    Get the class name for a concept from the configuration or by processing the concept name.
    
    Args:
        concept_name: Name of the concept
        config: Configuration dictionary
        
    Returns:
        Class name for the concept
    """
    # First check if there's a mapping in the config
    concept_to_class = config.get("concept_to_class", {})
    if concept_name in concept_to_class:
        return concept_to_class[concept_name]
    
    # Otherwise, process the concept name (remove underscores, etc.)
    return concept_name.replace("_", " ")


def verify_images(image_dir: str, expected_width: int = 1024, expected_height: int = 1024) -> bool:
    """
    Verify generated images have the correct dimensions and can be loaded.
    
    Args:
        image_dir: Directory containing the images
        expected_width: Expected width of the images
        expected_height: Expected height of the images
        
    Returns:
        True if all images are valid, False otherwise
    """
    image_files = list(Path(image_dir).glob("*.png"))
    
    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return False
    
    for image_file in image_files:
        try:
            img = Image.open(image_file)
            width, height = img.size
            
            if width != expected_width or height != expected_height:
                logger.warning(f"Image {image_file} has unexpected dimensions: {width}x{height}")
                return False
        except Exception as e:
            logger.warning(f"Failed to open image {image_file}: {e}")
            return False
    
    return True


if __name__ == "__main__":
    # This section can be used for testing the module directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate PEFT adapters on SDXL DreamBooth")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to pretrained SDXL model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained adapter weights")
    parser.add_argument("--concept_name", type=str, required=True, help="Name of the concept to evaluate")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to prompts file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images per prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--adapter_type", type=str, default=None, help="Type of adapter (loha, lora, etc.)")
    parser.add_argument("--rank", type=int, default=None, help="Rank of the adapter (if not auto-detected)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Prepare adapter config if manually specified
    adapter_config = None
    if args.adapter_type is not None:
        adapter_config = {
            "adapter_type": args.adapter_type,
            "rank": args.rank or 4,
            "alpha": args.rank or 4
        }
    
    # Load prompts
    prompt_templates, concept_to_class_map = load_prompts_from_file(args.prompts_file)
    
    # Render prompts
    rendered_prompts = render_prompt_templates(
        prompt_templates,
        unique_token="zjw",
        class_name=args.concept_name,
    )
    
    # Load pipeline with auto-detected configuration if not manually specified
    pipeline = load_inference_pipeline(
        pretrained_model_path=args.pretrained_model_path,
        adapter_path=args.adapter_path,
        adapter_name="test_adapter",
        adapter_config=adapter_config
    )
    
    # Generate samples for each prompt type
    all_images = []
    
    for prompt_type, prompt in rendered_prompts.items():
        prompt_output_dir = os.path.join(args.output_dir, prompt_type)
        
        images = generate_samples(
            pipeline=pipeline,
            prompts=prompt,
            output_dir=prompt_output_dir,
            num_images_per_prompt=args.num_images,
            seed=args.seed,
        )
        
        all_images.extend(images)
    
    # Verify generated images
    for prompt_type in rendered_prompts.keys():
        prompt_output_dir = os.path.join(args.output_dir, prompt_type)
        is_valid = verify_images(prompt_output_dir)
        logger.info(f"Images for {prompt_type} are valid: {is_valid}") 