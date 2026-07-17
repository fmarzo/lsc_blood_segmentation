"""
file: data_transforms.py

brief:
    This module defines training and evaluation transformations for HemoSet
    and the Rabbani Bleed Seg dataset.

    The training augmentation is designed to follow the augmentation strategy
    reported in the HemoSet paper as closely as possible:

    - random crops
    - brightness jitter

    The paper does not specify the exact crop size, crop probability,
    brightness range or brightness probability. Therefore, the numerical
    values used here are conservative implementation choices.

    Evaluation transformations are deterministic and do not include data
    augmentation.
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
# COMMON IMAGE SIZE
# ============================================================

IMAGE_SIZE = (
    480,
    640,
)


# ============================================================
# PAPER-LIKE AUGMENTATION PARAMETERS
# ============================================================

# The HemoSet paper reports random crops but does not specify their size.
# This interval preserves between 80% and 100% of the original image area.
CROP_SCALE = (
    0.80,
    1.00,
)

# Keep the original 4:3 aspect ratio.
CROP_RATIO = (
    4 / 3,
    4 / 3,
)

# The paper reports brightness jitter but does not specify its range.
# This moderate interval avoids excessively dark or bright samples.
BRIGHTNESS_RANGE = (
    0.80,
    1.20,
)

BRIGHTNESS_PROBABILITY = 0.80


def create_paper_like_train_transform():
    """
    Create the common paper-like training transformation.

    RandomResizedCrop performs a random spatial crop and resizes the result to
    the fixed network input resolution.

    When the dataset returns the segmentation mask as a torchvision
    tv_tensors.Mask, torchvision v2 automatically applies the same geometric
    transformation to image and mask while using nearest-neighbor
    interpolation for the mask.

    ColorJitter is applied only to the image and does not modify the mask.
    """
    return v2.Compose([

        # Randomly change framing, position and apparent blood size.
        v2.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=CROP_SCALE,
            ratio=CROP_RATIO,
            antialias=True,
        ),

        # Simulate moderate illumination differences between frames.
        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=BRIGHTNESS_RANGE,
                )
            ],
            p=BRIGHTNESS_PROBABILITY,
        ),

        # Convert image values to floating point in the [0, 1] range.
        # Mask class values remain integer labels.
        v2.ToDtype(
            torch.float32,
            scale=True,
        ),

        # Normalize RGB channels for the ImageNet-pretrained encoder.
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

    The output spatial resolution is [480, 640].
    """
    return create_paper_like_train_transform()


def create_eval_transform():
    """
    Create the deterministic HemoSet validation and test transformation.

    HemoSet images already have the expected spatial resolution, so no resize
    is applied.
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

    The same paper-like augmentation used for HemoSet is applied so that the
    two experiments use a comparable augmentation strategy.

    RandomResizedCrop directly produces images and masks with resolution
    [480, 640], independently of the original Rabbani image size.
    """
    return create_paper_like_train_transform()


def create_bleed_eval_transform():
    """
    Create the deterministic Rabbani Bleed Seg validation and test
    transformation.

    Rabbani images are resized to the common network resolution [480, 640].
    """
    return v2.Compose([

        v2.Resize(
            size=IMAGE_SIZE,
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