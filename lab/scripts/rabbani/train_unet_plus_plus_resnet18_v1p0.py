"""
file: train_unet_plus_plus_resnet18_v1p0.py

brief:
    Train a binary U-Net++ ResNet-18 model on the Rabbani Bleeding
    Segmentation v1.0 dataset.

    The configuration is fixed:

    - architecture: U-Net++
    - encoder: ResNet-18 pretrained on ImageNet
    - segmentation mode: binary
    - output channels: 1
    - foreground class: blood
    - loss: BCEWithLogitsLoss + DiceLoss
    - prediction threshold: 0.50
    - physical batch size: 4

    The training split receives Rabbani online augmentation.
    The validation split receives Rabbani evaluation preprocessing only.

    The best checkpoint is selected using validation blood Dice.

usage:
    python -u -m scripts.rabbani.train_unet_plus_plus_resnet18_v1p0 50

    If the number of epochs is omitted, config_split.DEFAULT_EPOCHS is used.
"""

import os
import random
import sys
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import (
    create_bleed_eval_transform,
    create_bleed_train_transform,
)
from src.hemoset_dataset_v2 import CustomImageDataset


# ============================================================
# FIXED MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "unet_plus_plus"
ENCODER_NAME = "resnet18"

SEGMENTATION_MODE = "binary"
NUM_OUTPUT_CHANNELS = 1
BLOOD_CLASS_INDEX = 0

BINARY_THRESHOLD = 0.50


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 4
NUM_WORKERS = 2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

SCHEDULER_MILESTONES = [10]
SCHEDULER_GAMMA = 0.1

RANDOM_SEED = 42
PRINT_EVERY_N_BATCHES = 25

EPSILON = 1e-12


# ============================================================
# DEVICE AND HARDWARE CONFIGURATION
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured for GPU training."
    )

DEVICE = torch.device("cuda")

# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings.
torch.backends.nnpack.set_flags(False)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def configure_reproducibility(seed):
    """
    Seed Python, NumPy and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """
    Give every DataLoader worker a reproducible random seed.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


configure_reproducibility(
    RANDOM_SEED
)


# ============================================================
# SEGMENTATION HELPERS
# ============================================================

def prepare_binary_mask(mask):
    """
    Convert a mask to binary float format while preserving [B, 1, H, W].

    This supports masks encoded either as 0/1 or 0/255.
    """
    return (
        mask > 0
    ).float().to(
        DEVICE,
        non_blocking=True,
    )


def compute_binary_loss(
    logits,
    mask,
    bce_loss,
    dice_loss,
):
    """
    Compute BCEWithLogitsLoss + DiceLoss.
    """
    return (
        bce_loss(
            logits,
            mask,
        )
        + dice_loss(
            logits,
            mask,
        )
    )


def get_binary_predictions(logits):
    """
    Convert logits to a binary blood prediction.
    """
    return (
        torch.sigmoid(logits)
        >= BINARY_THRESHOLD
    )


def validate_output_shape(
    logits,
    mask,
):
    """
    Verify that logits and masks have the same binary segmentation shape.
    """
    expected_shape = tuple(
        mask.shape
    )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected U-Net++ output shape. "
            f"Received {tuple(logits.shape)}, "
            f"expected {expected_shape}."
        )


def compute_batch_confusion(
    predictions,
    targets,
):
    """
    Compute TP, FP and FN independently for every image.
    """
    predictions = predictions.squeeze(
        dim=1
    )

    targets = targets.bool().squeeze(
        dim=1
    )

    true_positive = (
        predictions
        & targets
    ).sum(
        dim=(1, 2)
    ).double()

    false_positive = (
        predictions
        & ~targets
    ).sum(
        dim=(1, 2)
    ).double()

    false_negative = (
        ~predictions
        & targets
    ).sum(
        dim=(1, 2)
    ).double()

    return (
        true_positive,
        false_positive,
        false_negative,
    )


def compute_overlap_metrics(
    true_positive,
    false_positive,
    false_negative,
):
    """
    Compute per-image IoU and Dice.

    Empty target/prediction pairs receive a score of one.
    """
    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    per_image_iou = torch.where(
        iou_denominator > 0,
        true_positive
        / iou_denominator,
        torch.ones_like(
            iou_denominator
        ),
    )

    dice_denominator = (
        2.0 * true_positive
        + false_positive
        + false_negative
    )

    per_image_dice = torch.where(
        dice_denominator > 0,
        2.0
        * true_positive
        / dice_denominator,
        torch.ones_like(
            dice_denominator
        ),
    )

    return (
        per_image_iou,
        per_image_dice,
    )


# ============================================================
# NUMBER OF EPOCHS
# ============================================================

if len(sys.argv) > 1:
    n_epochs = int(
        sys.argv[1]
    )
else:
    n_epochs = getattr(
        config_split,
        "DEFAULT_EPOCHS",
        50,
    )

if n_epochs <= 0:
    raise ValueError(
        "The number of epochs must be greater than zero."
    )


# ============================================================
# RABBANI DATASET
# ============================================================

train_transform = (
    create_bleed_train_transform()
)

eval_transform = (
    create_bleed_eval_transform()
)


train_ds = CustomImageDataset(
    config_split.CSV_TRAIN_PATH_V1P0,
    train_transform,
)

valid_ds = CustomImageDataset(
    config_split.CSV_VALID_PATH_V1P0,
    eval_transform,
)


if len(train_ds) == 0:
    raise ValueError(
        "The Rabbani training dataset is empty."
    )

if len(valid_ds) == 0:
    raise ValueError(
        "The Rabbani validation dataset is empty."
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_generator = torch.Generator()
train_generator.manual_seed(
    RANDOM_SEED
)

validation_generator = torch.Generator()
validation_generator.manual_seed(
    RANDOM_SEED + 1
)


train_bleed_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    persistent_workers=(
        NUM_WORKERS > 0
    ),
    worker_init_fn=seed_worker,
    generator=train_generator,
)


valid_bleed_dl = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(
        NUM_WORKERS > 0
    ),
    worker_init_fn=seed_worker,
    generator=validation_generator,
)


if len(train_bleed_dl) == 0:
    raise ValueError(
        "The Rabbani training DataLoader is empty."
    )

if len(valid_bleed_dl) == 0:
    raise ValueError(
        "The Rabbani validation DataLoader is empty."
    )


# ============================================================
# MODEL
# ============================================================

model = smp.UnetPlusPlus(
    encoder_name=ENCODER_NAME,
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_OUTPUT_CHANNELS,
    activation=None,
).to(
    DEVICE
)


# ============================================================
# LOSS, OPTIMIZER AND SCHEDULER
# ============================================================

bce_loss = (
    torch.nn.BCEWithLogitsLoss()
    .to(DEVICE)
)

dice_loss = (
    smp.losses.DiceLoss(
        mode="binary",
        from_logits=True,
    )
    .to(DEVICE)
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=SCHEDULER_MILESTONES,
    gamma=SCHEDULER_GAMMA,
)


# ============================================================
# CHECKPOINT
# ============================================================

os.makedirs(
    config_split.MODEL_PRETRAINED_DIR,
    exist_ok=True,
)


checkpoint_path = os.path.join(
    config_split.MODEL_PRETRAINED_DIR,
    (
        "unet_plus_plus_binary_"
        "best_dice_bleed_seg_"
        f"{ENCODER_NAME}_rab.pth"
    ),
)


# ============================================================
# INITIAL SHAPE CHECK
# ============================================================

first_images, first_masks = next(
    iter(train_bleed_dl)
)


with torch.inference_mode():
    first_images = first_images.to(
        DEVICE,
        non_blocking=True,
    )

    first_masks = prepare_binary_mask(
        first_masks
    )

    first_logits = model(
        first_images
    )

    validate_output_shape(
        first_logits,
        first_masks,
    )


# ============================================================
# TRAINING SUMMARY
# ============================================================

print(
    "======== RABBANI U-NET++ BINARY TRAINING CONFIGURATION ========",
    flush=True,
)

print(
    f"Feature batch shape: {tuple(first_images.shape)}",
    flush=True,
)

print(
    f"Labels batch shape: {tuple(first_masks.shape)}",
    flush=True,
)

print(
    f"Model output shape: {tuple(first_logits.shape)}",
    flush=True,
)

print(
    f"Training CSV: {config_split.CSV_TRAIN_PATH_V1P0}",
    flush=True,
)

print(
    f"Validation CSV: {config_split.CSV_VALID_PATH_V1P0}",
    flush=True,
)

print(
    f"Training samples: {len(train_ds)}",
    flush=True,
)

print(
    f"Validation samples: {len(valid_ds)}",
    flush=True,
)

print(
    f"Training batches: {len(train_bleed_dl)}",
    flush=True,
)

print(
    f"Validation batches: {len(valid_bleed_dl)}",
    flush=True,
)

print(
    f"Model: {MODEL_NAME}",
    flush=True,
)

print(
    f"Encoder: {ENCODER_NAME}",
    flush=True,
)

print(
    f"Segmentation mode: {SEGMENTATION_MODE}",
    flush=True,
)

print(
    f"Output channels: {NUM_OUTPUT_CHANNELS}",
    flush=True,
)

print(
    "Training loss: BCEWithLogitsLoss + DiceLoss",
    flush=True,
)

print(
    f"Binary threshold: {BINARY_THRESHOLD:.2f}",
    flush=True,
)

print(
    f"Maximum epochs: {n_epochs}",
    flush=True,
)

print(
    f"Physical batch size: {BATCH_SIZE}",
    flush=True,
)

print(
    f"DataLoader workers: {NUM_WORKERS}",
    flush=True,
)

print(
    f"Learning rate: {LEARNING_RATE}",
    flush=True,
)

print(
    f"Scheduler milestones: {SCHEDULER_MILESTONES}",
    flush=True,
)

print(
    f"Checkpoint: {checkpoint_path}",
    flush=True,
)


del first_images
del first_masks
del first_logits


# ============================================================
# TRAINING STATE
# ============================================================

best_validation_dice = float(
    "-inf"
)

best_validation_loss = float(
    "inf"
)

best_epoch = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(n_epochs):
    epoch_start_time = time.perf_counter()

    print(
        f"\n------------ EPOCH: "
        f"{epoch + 1}/{n_epochs} "
        f"------------",
        flush=True,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss_sum = 0.0
    train_sample_count = 0

    training_start_time = time.perf_counter()


    for batch_index, (
        train_images,
        train_masks,
    ) in enumerate(train_bleed_dl):
        optimizer.zero_grad(
            set_to_none=True
        )

        train_images = train_images.to(
            DEVICE,
            non_blocking=True,
        )

        train_masks = prepare_binary_mask(
            train_masks
        )

        logits = model(
            train_images
        )

        if batch_index == 0:
            validate_output_shape(
                logits,
                train_masks,
            )

        loss_train_value = compute_binary_loss(
            logits=logits,
            mask=train_masks,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
        )

        if not torch.isfinite(
            loss_train_value
        ):
            raise FloatingPointError(
                "Non-finite training loss detected at "
                f"epoch {epoch + 1}, "
                f"batch {batch_index}."
            )

        loss_train_value.backward()

        optimizer.step()

        current_batch_size = (
            train_images.size(0)
        )

        train_loss_sum += (
            loss_train_value.item()
            * current_batch_size
        )

        train_sample_count += (
            current_batch_size
        )

        if (
            batch_index
            % PRINT_EVERY_N_BATCHES
            == 0
        ):
            print(
                f"train batch "
                f"{batch_index}/{len(train_bleed_dl)} "
                f"loss {loss_train_value.item():.6f}",
                flush=True,
            )


    avg_train_loss = (
        train_loss_sum
        / train_sample_count
    )

    training_duration = (
        time.perf_counter()
        - training_start_time
    )

    print(
        "Training phase completed in "
        f"{training_duration / 60.0:.2f} minutes.",
        flush=True,
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_sum = 0.0
    val_sample_count = 0
    val_image_count = 0

    global_true_positive = 0.0
    global_false_positive = 0.0
    global_false_negative = 0.0

    per_image_iou_sum = 0.0
    per_image_dice_sum = 0.0

    validation_start_time = time.perf_counter()

    print(
        f"Starting validation for epoch "
        f"{epoch + 1}/{n_epochs}...",
        flush=True,
    )


    with torch.inference_mode():
        for batch_index, (
            validation_images,
            validation_masks,
        ) in enumerate(valid_bleed_dl):
            validation_images = validation_images.to(
                DEVICE,
                non_blocking=True,
            )

            validation_masks = prepare_binary_mask(
                validation_masks
            )

            logits = model(
                validation_images
            )

            if batch_index == 0:
                validate_output_shape(
                    logits,
                    validation_masks,
                )

            loss_valid_value = compute_binary_loss(
                logits=logits,
                mask=validation_masks,
                bce_loss=bce_loss,
                dice_loss=dice_loss,
            )

            if not torch.isfinite(
                loss_valid_value
            ):
                raise FloatingPointError(
                    "Non-finite validation loss detected at "
                    f"epoch {epoch + 1}, "
                    f"batch {batch_index}."
                )

            predictions = get_binary_predictions(
                logits
            )

            (
                true_positive,
                false_positive,
                false_negative,
            ) = compute_batch_confusion(
                predictions,
                validation_masks,
            )

            (
                per_image_iou,
                per_image_dice,
            ) = compute_overlap_metrics(
                true_positive,
                false_positive,
                false_negative,
            )

            global_true_positive += float(
                true_positive.sum().item()
            )

            global_false_positive += float(
                false_positive.sum().item()
            )

            global_false_negative += float(
                false_negative.sum().item()
            )

            per_image_iou_sum += float(
                per_image_iou.sum().item()
            )

            per_image_dice_sum += float(
                per_image_dice.sum().item()
            )

            current_batch_size = (
                validation_images.size(0)
            )

            val_loss_sum += (
                loss_valid_value.item()
                * current_batch_size
            )

            val_sample_count += (
                current_batch_size
            )

            val_image_count += (
                current_batch_size
            )

            if (
                batch_index
                % PRINT_EVERY_N_BATCHES
                == 0
            ):
                print(
                    f"validation batch "
                    f"{batch_index}/{len(valid_bleed_dl)} "
                    f"loss {loss_valid_value.item():.6f}",
                    flush=True,
                )


    avg_val_loss = (
        val_loss_sum
        / val_sample_count
    )

    global_iou = (
        global_true_positive
        / (
            global_true_positive
            + global_false_positive
            + global_false_negative
            + EPSILON
        )
    )

    global_dice = (
        2.0
        * global_true_positive
        / (
            2.0 * global_true_positive
            + global_false_positive
            + global_false_negative
            + EPSILON
        )
    )

    global_precision = (
        global_true_positive
        / (
            global_true_positive
            + global_false_positive
            + EPSILON
        )
    )

    global_recall = (
        global_true_positive
        / (
            global_true_positive
            + global_false_negative
            + EPSILON
        )
    )

    mean_image_iou = (
        per_image_iou_sum
        / val_image_count
    )

    mean_image_dice = (
        per_image_dice_sum
        / val_image_count
    )

    validation_duration = (
        time.perf_counter()
        - validation_start_time
    )


    # ========================================================
    # CHECKPOINT SELECTION
    # ========================================================

    if global_dice > best_validation_dice:
        best_validation_dice = (
            global_dice
        )

        best_validation_loss = (
            avg_val_loss
        )

        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

        print(
            "New best checkpoint saved.",
            flush=True,
        )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step()

    current_learning_rate = (
        scheduler.get_last_lr()[0]
    )

    epoch_duration = (
        time.perf_counter()
        - epoch_start_time
    )


    # ========================================================
    # EPOCH SUMMARY
    # ========================================================

    print(
        f"\n----- Average values for epoch "
        f"{epoch + 1} -----",
        flush=True,
    )

    print(
        f"avg_train_loss: {avg_train_loss:.6f}",
        flush=True,
    )

    print(
        f"avg_val_loss: {avg_val_loss:.6f}",
        flush=True,
    )

    print(
        f"global_blood_iou: {global_iou:.6f}",
        flush=True,
    )

    print(
        f"global_blood_dice: {global_dice:.6f}",
        flush=True,
    )

    print(
        f"global_blood_precision: {global_precision:.6f}",
        flush=True,
    )

    print(
        f"global_blood_recall: {global_recall:.6f}",
        flush=True,
    )

    print(
        f"mean_image_iou: {mean_image_iou:.6f}",
        flush=True,
    )

    print(
        f"mean_image_dice: {mean_image_dice:.6f}",
        flush=True,
    )

    print(
        f"learning_rate: {current_learning_rate:.8f}",
        flush=True,
    )

    print(
        f"best_epoch: {best_epoch}",
        flush=True,
    )

    print(
        f"best_validation_dice: "
        f"{best_validation_dice:.6f}",
        flush=True,
    )

    print(
        "training_duration_minutes: "
        f"{training_duration / 60.0:.2f}",
        flush=True,
    )

    print(
        "validation_duration_minutes: "
        f"{validation_duration / 60.0:.2f}",
        flush=True,
    )

    print(
        "epoch_duration_minutes: "
        f"{epoch_duration / 60.0:.2f}",
        flush=True,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n======== RABBANI U-NET++ BINARY TRAINING COMPLETED ========",
    flush=True,
)

print(
    f"Best epoch: {best_epoch}",
    flush=True,
)

print(
    f"Best validation Dice: "
    f"{best_validation_dice:.6f}",
    flush=True,
)

print(
    f"Validation loss at best Dice: "
    f"{best_validation_loss:.6f}",
    flush=True,
)

print(
    f"Checkpoint saved to: {checkpoint_path}",
    flush=True,
)
