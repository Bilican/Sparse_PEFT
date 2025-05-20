#!/usr/bin/env python
# coding=utf-8

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from torchvision.transforms.functional import crop # <-- Import crop
import random
import itertools # <-- Import itertools for repeats

logger = logging.getLogger(__name__)

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for DreamBooth fine-tuning.
    Modified to correctly handle SDXL micro-conditioning (original_size, crop_top_left).
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer, # Tokenizer might not be strictly needed here if prompts are handled later
        instance_prompt=None,
        class_prompt=None,
        class_data_root=None,
        class_num=None,
        size=1024,
        center_crop=False,
        random_flip=False, # <-- Added random_flip argument for parity with Script 1
        encoder_hidden_states_dimension=2048, # May not be needed in dataset
        unique_token="sks",
        use_captions=False,
        caption_extension=".txt",
        repeats=1,
    ):
        """
        Initialize the DreamBooth dataset.

        Args:
            # ... (keep original args documentation) ...
            random_flip: Whether to randomly flip images horizontally during preprocessing.
            # ... (rest of args documentation) ...
        """
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer # Keep for potential future use, but not used in this modified version
        # self.encoder_hidden_states_dimension = encoder_hidden_states_dimension # Likely not needed here
        self.unique_token = unique_token
        self.use_captions = use_captions
        self.caption_extension = caption_extension
        self.random_flip = random_flip # Store the flag

        # --- Setup Prompts ---
        if instance_prompt is None:
            instance_prompt = f"a photo of {unique_token} object"
        self.instance_prompt = instance_prompt

        if class_prompt is None and instance_prompt is not None:
            class_prompt = instance_prompt.replace(f"{unique_token} ", "").replace(f"{unique_token}", "object") # More robust replace
        self.class_prompt = class_prompt

        # --- Load Instance Image Paths and Captions ---
        self.instance_images_path_raw = []
        self.instance_captions_raw = []
        if not os.path.isdir(instance_data_root):
             raise ValueError(f"Instance data root {instance_data_root} is not a valid directory.")
        
        instance_images_files = sorted(
            [f for f in os.listdir(instance_data_root) if self._check_image_file(f)],
            key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else x
        )

        if not instance_images_files:
            raise ValueError(f"No valid image files found in {instance_data_root}")

        for image_file in instance_images_files:
            image_path = os.path.join(instance_data_root, image_file)
            self.instance_images_path_raw.append(image_path)

            if use_captions:
                caption_path = os.path.join(
                    instance_data_root,
                    os.path.splitext(image_file)[0] + caption_extension
                )
                caption = self.instance_prompt # Default to instance prompt
                if os.path.exists(caption_path):
                    with open(caption_path, "r") as caption_file:
                        loaded_caption = caption_file.read().strip()
                        if loaded_caption: # Use caption only if not empty
                            caption = loaded_caption
                            # Ensure unique token is present if using captions
                            if unique_token not in caption:
                                logger.warning(f"Unique token '{unique_token}' not found in caption for {image_file}. Adding default prefix.")
                                # Simple prefix addition, adjust as needed
                                caption = f"a photo of {unique_token} {caption}"

                self.instance_captions_raw.append(caption)
            else:
                 # If not using captions, store None or the default prompt
                 # Storing the default prompt ensures list lengths match if needed later
                 self.instance_captions_raw.append(self.instance_prompt)


        logger.info(f"Found {len(self.instance_images_path_raw)} unique instance images.")

        # --- Preprocess Instance Images and Store Conditioning ---
        self.pixel_values = []
        self.original_sizes = []
        self.crop_top_lefts = []
        self.instance_prompts_processed = [] # Store the actual prompts used

        # Define transforms needed for preprocessing (similar to Script 1)
        # Note: ToTensor and Normalize are applied *after* recording crop coords
        _resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        _crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        _flip = transforms.RandomHorizontalFlip(p=1.0) # Will be applied conditionally
        _to_tensor = transforms.ToTensor()
        _normalize = transforms.Normalize([0.5], [0.5])

        # Repeat the raw paths/captions *before* processing
        instance_images_path_repeated = list(itertools.chain.from_iterable(itertools.repeat(x, repeats) for x in self.instance_images_path_raw))
        instance_captions_repeated = list(itertools.chain.from_iterable(itertools.repeat(x, repeats) for x in self.instance_captions_raw))


        logger.info(f"Processing {len(instance_images_path_repeated)} total instance images (including repeats)...")
        for i, image_path in enumerate(instance_images_path_repeated):
            try:
                image = Image.open(image_path)
                image = exif_transpose(image)

                if not image.mode == "RGB":
                    image = image.convert("RGB")

                # 1. Record original size
                self.original_sizes.append((image.height, image.width))

                # 2. Apply resize
                image = _resize(image)

                # 3. Optional random flip
                flipped = False
                if self.random_flip and random.random() < 0.5:
                    image = _flip(image)
                    flipped = True # You might need this info if transforms depend on it, but not for conditioning

                # 4. Get crop coordinates
                if center_crop:
                    y1 = max(0, int(round((image.height - size) / 2.0)))
                    x1 = max(0, int(round((image.width - size) / 2.0)))
                    h, w = size, size # Center crop yields target size
                else:
                     # Get random crop parameters for the current image size
                    y1, x1, h, w = _crop.get_params(image, (size, size))

                # Store crop coordinates
                self.crop_top_lefts.append((y1, x1))

                # 5. Apply crop
                image = crop(image, y1, x1, h, w)

                # 6. Apply ToTensor and Normalize
                img_tensor = _to_tensor(image)
                pixel_value = _normalize(img_tensor)
                self.pixel_values.append(pixel_value)

                # Store the corresponding prompt
                self.instance_prompts_processed.append(instance_captions_repeated[i])

            except Exception as e:
                 logger.error(f"Error processing instance image {image_path}: {e}. Skipping.")
                 # Remove potentially added partial data for this image to maintain list alignment
                 if len(self.original_sizes) > len(self.pixel_values): self.original_sizes.pop()
                 if len(self.crop_top_lefts) > len(self.pixel_values): self.crop_top_lefts.pop()
                 # We don't pop from instance_prompts_processed yet, handled below

        # Ensure all lists have the same length after skipping errors
        final_count = len(self.pixel_values)
        self.original_sizes = self.original_sizes[:final_count]
        self.crop_top_lefts = self.crop_top_lefts[:final_count]
        # Recreate prompts list based on successfully processed images
        # This requires tracking which original index succeeded, or simplifying:
        # We assume prompts list was appended correctly, just slice it if errors occurred.
        # This assumes the i-th prompt corresponds to the i-th successful image processing.
        self.instance_prompts_processed = self.instance_prompts_processed[:final_count]


        self.num_instance_images = len(self.pixel_values)
        if self.num_instance_images == 0:
             raise ValueError("No instance images were successfully processed. Check dataset and logs.")
        logger.info(f"Successfully processed {self.num_instance_images} instance images.")


        # --- Handle Class Images (Processing happens in __getitem__ for simplicity) ---
        self.class_images_path = []
        self.with_prior_preservation = False
        if class_data_root and class_num and class_prompt:
            if os.path.exists(class_data_root) and os.path.isdir(class_data_root):
                 class_image_files = sorted(
                    [f for f in os.listdir(class_data_root) if self._check_image_file(f)],
                     key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else x
                 )
                 if class_image_files:
                     self.with_prior_preservation = True
                     self.class_data_root = class_data_root # Store for use in getitem
                     # Store paths, actual loading/processing in getitem
                     self.class_images_path = [os.path.join(class_data_root, file) for file in class_image_files]
                     # Limit number of class images if needed
                     if len(self.class_images_path) > class_num:
                          # Sample randomly to avoid always using the first N
                          self.class_images_path = random.sample(self.class_images_path, class_num)
                     self.num_class_images = len(self.class_images_path)
                     logger.info(f"Found {self.num_class_images} class images for prior preservation.")
                 else:
                     logger.warning(f"No valid image files found in class data directory {class_data_root}. Prior preservation disabled.")
            else:
                 logger.warning(f"Class data directory {class_data_root} not found or not a directory. Prior preservation disabled.")
        else:
            logger.info("Prior preservation not configured.")

        # Define the transforms to be applied to class images within __getitem__
        # We need resize, crop, ToTensor, Normalize
        self.class_image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                # We will manually get crop params for class images in getitem
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # Store the crop transform separately to get params
        self.class_crop_transform = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)

        self._length = self.num_instance_images # Length is determined by instance images


    def _check_image_file(self, filename):
        """Check if file is a supported image format."""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

    def __len__(self):
        # The length of the dataset is based on the preprocessed instance images
        return self._length

    def __getitem__(self, index):
        example = {}

        # --- Instance Data (Preprocessed) ---
        example["instance_images"] = self.pixel_values[index % self.num_instance_images] # Use modulo for safety
        example["original_size"] = self.original_sizes[index % self.num_instance_images]
        example["crop_top_left"] = self.crop_top_lefts[index % self.num_instance_images]
        example["instance_prompt"] = self.instance_prompts_processed[index % self.num_instance_images]

        # --- Class Data (Processed on the fly) ---
        if self.with_prior_preservation:
            class_idx = index % self.num_class_images # Cycle through class images
            class_image_path = self.class_images_path[class_idx]
            try:
                class_image = Image.open(class_image_path)
                class_image = exif_transpose(class_image)

                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")

                # Calculate conditioning for the class image
                class_original_size = (class_image.height, class_image.width)

                # Apply resize (part of self.class_image_transforms)
                # Need intermediate image after resize to calculate crop
                _resize = self.class_image_transforms.transforms[0]
                resized_class_image = _resize(class_image)

                # Apply optional flip *before* getting crop params for consistency
                flipped = False
                if self.random_flip and random.random() < 0.5:
                     # Use the same flip instance defined earlier
                     _flip = transforms.RandomHorizontalFlip(p=1.0)
                     resized_class_image = _flip(resized_class_image)
                     flipped = True

                # Get crop params for the (potentially flipped) resized class image
                if self.center_crop:
                    y1 = max(0, int(round((resized_class_image.height - self.size) / 2.0)))
                    x1 = max(0, int(round((resized_class_image.width - self.size) / 2.0)))
                    h, w = self.size, self.size
                else:
                    y1, x1, h, w = self.class_crop_transform.get_params(resized_class_image, (self.size, self.size))

                class_crop_top_left = (y1, x1)

                # Apply crop
                cropped_class_image = crop(resized_class_image, y1, x1, h, w)

                # Apply remaining transforms (ToTensor, Normalize)
                class_pixel_values = self.class_image_transforms.transforms[1](cropped_class_image) # ToTensor
                class_pixel_values = self.class_image_transforms.transforms[2](class_pixel_values) # Normalize

                example["class_images"] = class_pixel_values
                example["class_original_size"] = class_original_size
                example["class_crop_top_left"] = class_crop_top_left
                example["class_prompt"] = self.class_prompt

            except Exception as e:
                 logger.error(f"Error processing class image {class_image_path} in __getitem__: {e}. Skipping class data for this example.")
                 # Ensure keys exist even if processing fails, potentially with None or default values
                 example["class_images"] = torch.zeros((3, self.size, self.size)) # Placeholder tensor
                 example["class_original_size"] = (self.size, self.size) # Placeholder
                 example["class_crop_top_left"] = (0, 0) # Placeholder
                 example["class_prompt"] = self.class_prompt # Still provide prompt

        # If not using prior preservation, these keys won't exist, which is fine.
        # The collate function needs to handle potentially missing keys if class processing fails.
        return example

# --- Collate Function (Modified to handle potential missing class keys) ---

def collate_fn(examples, with_prior_preservation=False):
    """
    Collate function for dataloader to prepare batch of examples.
    Modified to be robust to missing class data if processing failed.
    """
    # Instance data is always present
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]

    # Initialize lists for potential class data
    class_pixel_values = []
    class_prompts = []
    class_original_sizes = []
    class_crop_top_lefts = []

    if with_prior_preservation:
        for example in examples:
            # Only append class data if it was successfully processed and added in __getitem__
            if "class_images" in example:
                class_pixel_values.append(example["class_images"])
                class_prompts.append(example["class_prompt"])
                class_original_sizes.append(example["class_original_size"])
                class_crop_top_lefts.append(example["class_crop_top_left"])
            else:
                # Handle cases where class image processing might have failed in __getitem__
                # Option 1: Skip adding class data for this example (might imbalance batch)
                # Option 2: Add placeholder data (might affect loss calculation if not handled carefully)
                # Let's log a warning and potentially add placeholders if needed,
                # but for now, we assume __getitem__ provides defaults if needed.
                # If class_prompts becomes shorter than prompts, need careful handling later.
                 logger.warning("Missing class data in collate_fn for one example.")
                 # To keep lists aligned for simple concatenation, add placeholders
                 # Ensure placeholders match expected types/shapes if used
                 # class_pixel_values.append(torch.zeros_like(pixel_values[0])) # Example placeholder
                 # class_prompts.append("")
                 # class_original_sizes.append((0,0))
                 # class_crop_top_lefts.append((0,0))
                 # For simplicity now, we assume __getitem__ added defaults if error occurred.


        # Concatenate class and instance examples only if class data is available
        if class_pixel_values: # Check if any class data was actually collected
             pixel_values.extend(class_pixel_values)
             prompts.extend(class_prompts)
             original_sizes.extend(class_original_sizes)
             crop_top_lefts.extend(class_crop_top_lefts)


    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }

    return batch


def create_dataloader(
    dataset,
    batch_size=1,
    shuffle=True,
    with_prior_preservation=False,
    num_workers=0
):
    """
    Create a dataloader from the provided dataset.
    """
    # Ensure the dataset's prior preservation status matches the dataloader's
    effective_prior_preservation = with_prior_preservation and dataset.with_prior_preservation

    if with_prior_preservation and not dataset.with_prior_preservation:
        logger.warning("Dataloader created with_prior_preservation=True, but dataset has no class images. Disabling.")
        effective_prior_preservation = False


    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda examples: collate_fn(examples, effective_prior_preservation),
        num_workers=num_workers,
    )