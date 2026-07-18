"""
file: data_transforms.py

brief:
    This module defines training and evaluation transformations for HemoSet
    and the Rabbani Bleed Seg dataset.

    HemoSet images retain their original rectified resolution of 640x480
    pixels, represented in torchvision as (height, width) = (480, 640).

    The Rabbani paper reports a resolution of 854x480 pixels. Rabbani images
    are therefore transformed to (480, 864), which preserves approximately
    the original 16:9 aspect ratio while making both dimensions divisible
    by 32 for the U-Net encoder-decoder path.

    The HemoSet training augmentation follows the operations reported in the
    HemoSet paper:

    - random crops
    - brightness jitter

    The paper does not specify the exact crop range, brightness range or
    application probability. The numerical values used here are conservative
    implementation choices.

    The same crop and brightness augmentation is currently used for Rabbani
    to keep the image-based experiments comparable. Evaluation transforms are
    deterministic and do not include augmentation.
"""

import torch

from torchvision.transforms import v2


# ============================================================
# IMAGENET NORMALIZATION
# ============================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


# ============================================================
# DATASET-SPECIFIC IMAGE SIZES
# ============================================================

# HemoSet is rectified to 640x480 pixels.
# Torchvision uses the order (height, width).
HEMOSET_IMAGE_SIZE = (
    480,
    640,
)

# The Rabbani paper reports 854x480 pixels.
# Width 864 is used because it is close to 854 and divisible by 32.
RABBANI_IMAGE_SIZE = (
    480,
    864,
)


# ============================================================
# AUGMENTATION PARAMETERS
# ============================================================

# Preserve between 80% and 100% of the original image area.
CROP_SCALE = (
    0.80,
    1.00,
)

# Preserve the original HemoSet 4:3 aspect ratio.
HEMOSET_CROP_RATIO = (
    HEMOSET_IMAGE_SIZE[1] / HEMOSET_IMAGE_SIZE[0],
    HEMOSET_IMAGE_SIZE[1] / HEMOSET_IMAGE_SIZE[0],
)

# Preserve approximately the original Rabbani 16:9 aspect ratio.
RABBANI_CROP_RATIO = (
    RABBANI_IMAGE_SIZE[1] / RABBANI_IMAGE_SIZE[0],
    RABBANI_IMAGE_SIZE[1] / RABBANI_IMAGE_SIZE[0],
)

# Moderate brightness jitter.
BRIGHTNESS_RANGE = (
    0.80,
    1.20,
)

BRIGHTNESS_PROBABILITY = 0.80


def create_crop_brightness_train_transform(
    image_size,
    crop_ratio,
):
    """
    Create a training transformation using random crop and brightness jitter.

    RandomResizedCrop applies the same geometric transformation to the image
    and mask when the mask is represented as torchvision.tv_tensors.Mask.

    Torchvision v2 automatically uses the appropriate interpolation for the
    segmentation mask, preserving its discrete class values.

    Brightness jitter is applied only to the input image and does not modify
    the segmentation mask.
    """
    return v2.Compose([

        # Randomly modify framing, position and apparent blood size.
        v2.RandomResizedCrop(
            size=image_size,
            scale=CROP_SCALE,
            ratio=crop_ratio,
            antialias=True,
        ),

        # Simulate moderate illumination differences.
        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=BRIGHTNESS_RANGE,
                )
            ],
            p=BRIGHTNESS_PROBABILITY,
        ),

        # Convert image values to floating point in the [0, 1] range.
        # Segmentation mask class values remain unchanged.
        v2.ToDtype(
            torch.float32,
            scale=True,
        ),

        # Normalize the RGB image for the ImageNet-pretrained encoder.
        v2.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

        v2.ToPureTensor(),
    ])


# ============================================================
# HEMOSET TRANSFORMS
# ============================================================

def create_train_transform():
    """
    Create the HemoSet training transformation.

    Output tensor spatial resolution:
        height = 480
        width = 640
    """
    return create_crop_brightness_train_transform(
        image_size=HEMOSET_IMAGE_SIZE,
        crop_ratio=HEMOSET_CROP_RATIO,
    )


def create_eval_transform():
    """
    Create the deterministic HemoSet validation and test transformation.

    HemoSet images are already stored at the expected 640x480 resolution,
    so no resizing is applied.
    """
    return v2.Compose([

        v2.ToDtype(
            torch.float32,
            scale=True,
        ),

        v2.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

        v2.ToPureTensor(),
    ])


# ============================================================
# RABBANI BLEED SEG TRANSFORMS
# ============================================================

def create_bleed_train_transform():
    """
    Create the Rabbani Bleed Seg training transformation.

    RandomResizedCrop preserves an aspect ratio close to the 854x480
    resolution reported in the Rabbani paper.

    Output tensor spatial resolution:
        height = 480
        width = 864
    """
    return create_crop_brightness_train_transform(
        image_size=RABBANI_IMAGE_SIZE,
        crop_ratio=RABBANI_CROP_RATIO,
    )


def create_bleed_eval_transform():
    """
    Create the deterministic Rabbani validation and test transformation.

    Rabbani images and masks are resized to 480x864, preserving approximately
    the aspect ratio reported in the paper.
    """
    return v2.Compose([

        v2.Resize(
            size=RABBANI_IMAGE_SIZE,
            antialias=True,
        ),

        v2.ToDtype(
            torch.float32,
            scale=True,
        ),

        v2.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),

        v2.ToPureTensor(),
    ])