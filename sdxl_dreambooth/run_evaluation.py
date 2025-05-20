#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import logging
import yaml
import torch
import sys # Import sys to exit
from pathlib import Path
from glob import glob
import json

from sdxl_dreambooth.evaluation import batch_evaluation, detect_adapter_config, load_prompts_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation for PEFT adapters on SDXL DreamBooth")
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/experiment_config.yaml",
        help="Path to the experiment configuration file",
    )
    parser.add_argument(
        "--adapter_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Directories containing trained adapters. If not provided, will look for adapters in the output directory.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=None,
        help="List of concepts to evaluate. If not provided, will use all concepts in the dataset directory.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to the pretrained SDXL model. Overrides the value in the config file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs. Overrides the value in the config file.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to the prompts file. Overrides the value in the config file.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=3,
        help="Maximum number of prompts to use (randomly selected). If not provided, all prompts are used.",
    )
    parser.add_argument(
        "--use_unique_seeds",
        action="store_true",
        help="Use different random seeds for each image generated with the same prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="fp16" if torch.cuda.is_available() else "no",
        help="Mixed precision mode for evaluation",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Specific adapter method to evaluate (e.g., 'loha', 'lora'). If not provided, all found adapters will be evaluated.",
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        help="Save a grid of all generated images for easier comparison",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--generate_baseline",
        action="store_true",
        help="Generate baseline results (without adapter) for comparison",
    )
    parser.add_argument(
        "--unique_token",
        type=str,
        default="zjw",
        help="Unique token to use in prompts (replaces the default in experiment_config.yaml)",
    )
    
    return parser.parse_args()


def find_adapter_paths(adapter_dirs, method_filter=None, generate_baseline=False):
    """
    Find paths to adapter checkpoints in the given directories.
    
    Args:
        adapter_dirs: List of directories to search for adapters
        method_filter: Optional adapter method to filter by (e.g., 'loha', 'lora')
        generate_baseline: Whether to include baseline results (without adapter)
        
    Returns:
        Dictionary mapping method names to adapter paths
    """
    adapter_paths = {}
    
    # Add a baseline method (no adapter) if requested
    if generate_baseline:
        adapter_paths["baseline"] = None
        logger.info("Adding baseline method (no adapter) for comparison")
    
        if method_filter and method_filter.lower() != "baseline":
            logger.info(f"Removing baseline as it doesn't match the requested method {method_filter}")
            del adapter_paths["baseline"]
    
    logger.info(f"Searching for adapters in directories: {adapter_dirs}")
    
    # Make sure we have valid directories
    valid_dirs = []
    for adapter_dir in adapter_dirs:
        if os.path.isdir(adapter_dir):
            valid_dirs.append(adapter_dir)
        else:
            logger.warning(f"Directory not found: {adapter_dir}")
    
    if not valid_dirs:
        logger.warning("No valid adapter directories found. Using absolute paths as fallback.")
        # Try absolute paths as fallback
        absolute_dirs = [
            "/home/codeway/Peft-League/outputs",
            "/home/codeway/Peft-League/outputs/final_model"
        ]
        for abs_dir in absolute_dirs:
            if os.path.isdir(abs_dir):
                valid_dirs.append(abs_dir)
                logger.info(f"Found fallback directory: {abs_dir}")
    
    for adapter_dir in valid_dirs:
        logger.info(f"Searching for adapters in {adapter_dir}")
        
        # First check for adapter_config.json to determine adapter type
        config_files = glob(os.path.join(adapter_dir, "adapter_config.json"))
        config_files.extend(glob(os.path.join(adapter_dir, "**/adapter_config.json")))
        
        adapter_configs = {}
        
        for config_file in config_files:
            logger.info(f"Found adapter config file: {config_file}")
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)
                    if "peft_type" in config_data:
                        peft_type = config_data["peft_type"].upper()
                        if peft_type == "LOHA":
                            adapter_type = "loha"
                        elif peft_type == "LORA":
                            adapter_type = "lora"
                        else:
                            adapter_type = peft_type.lower()
                        
                        # Store the config for this directory
                        config_dir = os.path.dirname(config_file)
                        adapter_configs[config_dir] = {
                            "type": adapter_type,
                            "path": config_file
                        }
                        logger.info(f"Found explicit adapter type in config: {adapter_type}")
            except Exception as e:
                logger.warning(f"Error reading adapter config: {str(e)}")
        
        # Search for model files in the current directory and subdirectories
        adapter_files = []
        
        # File patterns to search for
        file_patterns = ["*.safetensors", "*.pt", "*.bin", "*.ckpt", "model.*"]
        
        # Look in the current directory
        for pattern in file_patterns:
            files = glob(os.path.join(adapter_dir, pattern))
            if files:
                logger.debug(f"Found {len(files)} files matching {pattern} in {adapter_dir}")
                adapter_files.extend(files)
        
        # Look in common subdirectories
        common_subdirs = ["final_model", "checkpoints", "adapter", "models", "adapters"]
        for subdir in common_subdirs:
            subdir_path = os.path.join(adapter_dir, subdir)
            if os.path.isdir(subdir_path):
                for pattern in file_patterns:
                    files = glob(os.path.join(subdir_path, pattern))
                    if files:
                        logger.debug(f"Found {len(files)} files matching {pattern} in {subdir_path}")
                        adapter_files.extend(files)
                    
                # Check for adapter directory inside this subdir
                adapter_subdir = os.path.join(subdir_path, "adapter")
                if os.path.isdir(adapter_subdir):
                    for pattern in file_patterns:
                        files = glob(os.path.join(adapter_subdir, pattern))
                        if files:
                            logger.debug(f"Found {len(files)} files matching {pattern} in {adapter_subdir}")
                            adapter_files.extend(files)
        
        # Also search in all subdirectories one level deep
        for subdir in [d for d in glob(os.path.join(adapter_dir, "*")) if os.path.isdir(d)]:
            # Skip directories we've already checked or that are typically not relevant
            if os.path.basename(subdir) in common_subdirs + ["logs", "config", "samples", "evaluation"]:
                continue
            
            for pattern in file_patterns:
                files = glob(os.path.join(subdir, pattern))
                if files:
                    logger.debug(f"Found {len(files)} files matching {pattern} in {subdir}")
                    adapter_files.extend(files)
        
        logger.info(f"Found {len(adapter_files)} potential adapter files in {adapter_dir} and subdirectories")
        
        # Filter out obvious non-adapter files
        excluded_keywords = ["optimizer", "scheduler", "random", ".tmp", ".DS_Store"]
        filtered_files = []
        for file in adapter_files:
            if not any(keyword in os.path.basename(file).lower() for keyword in excluded_keywords):
                filtered_files.append(file)
        
        logger.info(f"After filtering, {len(filtered_files)} potential adapter files remain")
        
        # Process found adapter files
        for adapter_file in filtered_files:
            try:
                logger.info(f"Analyzing potential adapter file: {adapter_file}")
                
                # Check if there's a config file in the same directory first
                adapter_file_dir = os.path.dirname(adapter_file)
                
                # Check if we have a known config for this directory
                if adapter_file_dir in adapter_configs:
                    config_info = adapter_configs[adapter_file_dir]
                    method_name = config_info["type"]
                    logger.info(f"Using adapter type {method_name} from config file {config_info['path']}")
                else:
                    # Fallback to detecting from the file itself
                    adapter_config = detect_adapter_config(adapter_file)
                    method_name = adapter_config["adapter_type"]
                    logger.info(f"Detected adapter type {method_name} from file content")
                
                # If a method filter is provided, only include matching adapters
                if method_filter and method_filter.lower() != method_name.lower():
                    logger.info(f"Skipping {adapter_file} as it doesn't match the requested method {method_filter}")
                    continue
                
                # Use a more descriptive name if possible (from the directory structure)
                parent_dir = os.path.basename(os.path.dirname(adapter_file))
                if parent_dir not in ["final_model", "checkpoints", "adapter", "adapters"] and not parent_dir.startswith("."):
                    method_name = f"{parent_dir}_{method_name}"
                
                # Check for duplicates and take the most recent
                if method_name in adapter_paths:
                    current_time = os.path.getmtime(adapter_paths[method_name])
                    new_time = os.path.getmtime(adapter_file)
                    if new_time > current_time:
                        logger.info(f"Replacing {adapter_paths[method_name]} with newer file {adapter_file}")
                        adapter_paths[method_name] = adapter_file
                else:
                    adapter_paths[method_name] = adapter_file
                    logger.info(f"Found adapter for {method_name} at {adapter_file}")
            except Exception as e:
                logger.warning(f"Could not process adapter file {adapter_file}: {e}")
    
    if len(adapter_paths) <= 1:  # Just the baseline
        logger.warning("No adapters found. Looking for specific model files.")
        # Try to find any model file as a last resort
        for adapter_dir in valid_dirs:
            model_file = os.path.join(adapter_dir, "model.safetensors")
            if os.path.exists(model_file):
                method_name = "unknown"
                try:
                    # Check for adapter_config.json in the same directory
                    config_file = os.path.join(os.path.dirname(model_file), "adapter_config.json")
                    if os.path.exists(config_file):
                        with open(config_file, "r") as f:
                            config_data = json.load(f)
                            if "peft_type" in config_data:
                                peft_type = config_data["peft_type"].upper()
                                if peft_type == "LOHA":
                                    method_name = "loha"
                                elif peft_type == "LORA":
                                    method_name = "lora"
                                else:
                                    method_name = peft_type.lower()
                    else:
                        # Fallback to detecting from file
                        adapter_config = detect_adapter_config(model_file)
                        method_name = adapter_config["adapter_type"]
                except Exception as e:
                    logger.warning(f"Error detecting adapter type: {e}")
                
                adapter_paths[method_name] = model_file
                logger.info(f"Found adapter at {model_file} as a last resort")
                break
    
    return adapter_paths


def find_concepts(dataset_dir, concepts=None):
    """
    Find concepts in the dataset directory.
    
    Args:
        dataset_dir: Path to the dataset directory
        concepts: Optional list of specific concepts to use
        
    Returns:
        List of concept names
    """
    # If specific concepts are provided, use them without checking directories
    if concepts and len(concepts) > 0:
        return concepts
    
    # Check multiple possible dataset locations
    possible_dirs = [
        dataset_dir,
        os.path.join(os.getcwd(), "dataset"),
        os.path.join(os.getcwd(), "../dataset"),
        "/home/codeway/Peft-League/dataset"
    ]
    
    # Try each possible directory
    for dir_path in possible_dirs:
        if os.path.isdir(dir_path):
            logger.info(f"Checking for concepts in {dir_path}")
            # Find all subdirectories in the dataset directory
            concept_dirs = [
                d for d in glob(os.path.join(dir_path, "*"))
                if os.path.isdir(d) and not os.path.basename(d).startswith(".")
            ]
            
            if concept_dirs:
                concept_names = [os.path.basename(d) for d in concept_dirs]
                logger.info(f"Found {len(concept_names)} concepts in {dir_path}: {concept_names}")
                return concept_names
    
    # If no concepts found in any directory, use default concepts
    logger.warning("No concept directories found. Using default concept 'dog'.")
    return ["dog"]


def main():
    args = parse_args()
    
    # Load config file
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config values with command-line arguments
    if args.pretrained_model_path:
        config["pretrained_model_path"] = args.pretrained_model_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.prompts_file:
        config["prompts_file"] = args.prompts_file
    # Override unique token if provided
    if args.unique_token:
        config["unique_token"] = args.unique_token
        logger.info(f"Using custom unique token: {args.unique_token}")
    
    # Set up paths
    pretrained_model_path = config["pretrained_model_path"]
    
    # Try multiple potential dataset directories
    dataset_dir = None
    potential_dataset_dirs = [
        config.get("dataset_dir"),
        "dataset",
        "../dataset",
        "/home/codeway/Peft-League/dataset"
    ]
    
    for potential_dir in potential_dataset_dirs:
        if potential_dir and os.path.isdir(potential_dir):
            dataset_dir = potential_dir
            logger.info(f"Using dataset directory: {dataset_dir}")
            break
    
    if not dataset_dir:
        logger.warning("Could not find dataset directory. Using default path 'dataset'")
        dataset_dir = "dataset"
    
    output_base_dir = os.path.join(config["output_dir"], "evaluation")
    
    # Handle prompts file path
    prompts_file = args.prompts_file if args.prompts_file else config.get("prompts_file")
    
    # Try different locations for prompts file if not specified or doesn't exist
    if not prompts_file or not os.path.exists(prompts_file):
        potential_prompts_files = [
            os.path.join(dataset_dir, "prompts_and_classes.txt"),
            "dataset/prompts_and_classes.txt",
            "../dataset/prompts_and_classes.txt",
            "/home/codeway/Peft-League/dataset/prompts_and_classes.txt"
        ]
        
        for potential_file in potential_prompts_files:
            if os.path.exists(potential_file):
                prompts_file = potential_file
                logger.info(f"Using prompts file: {prompts_file}")
                break
        
        if not prompts_file or not os.path.exists(prompts_file):
            logger.error("No prompts file found. Please provide a valid prompts file.")
            return
    
    # Load prompts and class map
    prompt_templates, concept_to_class_map = load_prompts_from_file(prompts_file)

    # --- Critical Check: Ensure prompts were loaded ---
    if not prompt_templates:
        logger.critical(f"Failed to load or parse any prompts from {prompts_file}. Check the file format and logs. Exiting.")
        sys.exit(1) # Exit with error code
    else:
        logger.info(f"Successfully loaded {len(prompt_templates)} prompt templates.")

    if not concept_to_class_map:
        logger.warning(f"No class mappings were found or parsed in {prompts_file}. Class names will be derived from concept names.")
    else:
        logger.info(f"Successfully loaded {len(concept_to_class_map)} concept-to-class mappings.")

    # Set up device and precision
    device = args.device
    torch_dtype = torch.float32
    if args.mixed_precision == "fp16":
        torch_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16
    
    # Find adapter paths
    adapter_dirs = args.adapter_dirs or [config["output_dir"]]
    adapter_paths = find_adapter_paths(adapter_dirs, args.method, args.generate_baseline)
    
    if not adapter_paths:
        logger.error("No adapter checkpoints found. Please check the adapter directories.")
        return
    
    # Find concepts
    concept_names = find_concepts(dataset_dir, args.concepts)
    
    logger.info(f"Found {len(adapter_paths)} adapters: {list(adapter_paths.keys())}")
    logger.info(f"Using {len(concept_names)} concepts: {concept_names}")
    
    # Extra parameters for evaluation
    evaluation_params = {
        "max_prompts": args.max_prompts,
        "use_unique_seeds": args.use_unique_seeds,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps
    }
    
    # Run batch evaluation
    batch_evaluation(
        config=config,
        adapter_paths=adapter_paths,
        concept_names=concept_names,
        prompts_file=prompts_file,
        output_base_dir=output_base_dir,
        pretrained_model_path=pretrained_model_path,
        device=device,
        num_images_per_prompt=args.num_images,
        torch_dtype=torch_dtype,
        seed=args.seed,
        save_grid=args.save_grid,
        evaluation_params=evaluation_params
    )
    
    logger.info("Evaluation complete. Results saved to {}".format(output_base_dir))


if __name__ == "__main__":
    main() 