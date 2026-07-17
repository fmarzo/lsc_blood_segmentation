"""
file: train_rabbani.py

brief:
    This script trains a U-Net model on the Rabbani Bleed Seg dataset.

    The architecture is fixed to U-Net with a ResNet-18 encoder pretrained
    on ImageNet.

    The segmentation task is fixed to multiclass segmentation with two output
    classes:

    - class 0: background
    - class 1: blood

    The training set uses random crop, horizontal flip and color augmentation.
    The validation set uses only deterministic resizing and normalization.

    The checkpoint is selected according to the highest validation Dice score
    of the blood class.

usage:
    python -m scripts.train_rabbani 50

    If the number of epochs is omitted, config_split.DEFAULT_EPOCHS is used.
"""

import os
import sys

import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import (
    create_bleed_train_transform,
    create_bleed_eval_transform,
)
from src.hemoset_dataset_v2 import CustomImageDataset


# ============================================================
# FIXED RABBANI TRAINING CONFIGURATION
# ============================================================

MODEL_NAME = "unet"
ENCODER_NAME = "resnet18"

SEGMENTATION_MODE = "multiclass"
NUM_CLASSES = 2

BACKGROUND_CLASS_INDEX = 0
BLOOD_CLASS_INDEX = 1

BATCH_SIZE = 4
NUM_WORKERS = 2

LEARNING_RATE = 0.001
LR_MILESTONES = [10]
LR_GAMMA = 0.1


# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings on the CPU node.
torch.backends.nnpack.set_flags(False)


def prepare_mask(mask):
    """
    Convert the mask from [B, 1, H, W] to [B, H, W] and prepare it for
    CrossEntropyLoss.
    """
    return torch.squeeze(
        mask,
        dim=1,
    ).long().to(
        "cuda",
        non_blocking=True,
    )


def get_predictions(logits):
    """
    Convert multiclass logits into the predicted class map.
    """
    return logits.argmax(
        dim=1,
    )


def get_segmentation_stats(predictions, mask):
    """
    Compute TP, FP, FN and TN separately for every class.
    """
    return smp.metrics.get_stats(
        predictions,
        mask,
        mode=SEGMENTATION_MODE,
        num_classes=NUM_CLASSES,
    )


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured for GPU training."
    )


# Read the number of epochs from the command line.
if len(sys.argv) > 1:
    n_epochs = int(sys.argv[1])
else:
    n_epochs = config_split.DEFAULT_EPOCHS


if n_epochs <= 0:
    raise ValueError(
        "The number of epochs must be greater than zero."
    )


# ============================================================
# DATASET AND TRANSFORMS
# ============================================================

train_transform = create_bleed_train_transform()
eval_transform = create_bleed_eval_transform()


train_ds = CustomImageDataset(
    config_split.CSV_TRAIN_PATH_V1P0,
    train_transform,
)


valid_ds = CustomImageDataset(
    config_split.CSV_VALID_PATH_V1P0,
    eval_transform,
)


train_bleed_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=True,
)


valid_bleed_dl = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
)


if len(train_bleed_dl) == 0:
    raise ValueError(
        "The Rabbani training DataLoader is empty."
    )


if len(valid_bleed_dl) == 0:
    raise ValueError(
        "The Rabbani validation DataLoader is empty."
    )


# Inspect the shape of the first training batch.
train_img, train_mask = next(
    iter(train_bleed_dl)
)


print("======== RABBANI TRAINING CONFIGURATION ========")

print(
    f"Feature batch shape: {train_img.size()}"
)

print(
    f"Labels batch shape: {train_mask.size()}"
)

print(
    f"Training samples: {len(train_ds)}"
)

print(
    f"Validation samples: {len(valid_ds)}"
)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Encoder: {ENCODER_NAME}"
)

print(
    f"Segmentation mode: {SEGMENTATION_MODE}"
)

print(
    f"Number of classes: {NUM_CLASSES}"
)

print(
    f"Epochs: {n_epochs}"
)


# ============================================================
# MODEL
# ============================================================

model = smp.Unet(
    encoder_name=ENCODER_NAME,
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to("cuda")


# ============================================================
# OPTIMIZER AND SCHEDULER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=LR_MILESTONES,
    gamma=LR_GAMMA,
)


# Multiclass loss.
loss_function = torch.nn.CrossEntropyLoss().to(
    "cuda"
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
        "unet_multiclass_"
        "best_dice_bleed_seg_"
        "resnet18.pth"
    ),
)


best_val_dice = float("-inf")
best_epoch = 0


print(
    f"Checkpoint: {checkpoint_path}"
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(n_epochs):

    print(
        f"\n------------ EPOCH: "
        f"{epoch + 1}/{n_epochs} ------------"
    )

    model.train()

    train_loss_sum = 0.0
    train_sample_count = 0


    for batch_index, (
        train_img,
        train_mask,
    ) in enumerate(train_bleed_dl):

        optimizer.zero_grad(
            set_to_none=True
        )

        train_img = train_img.to(
            "cuda",
            non_blocking=True,
        )

        train_mask = prepare_mask(
            train_mask
        )

        logits = model(
            train_img
        )

        loss_train_value = loss_function(
            logits,
            train_mask,
        )

        loss_train_value.backward()

        optimizer.step()


        current_batch_size = train_img.size(0)

        train_loss_sum += (
            loss_train_value.item()
            * current_batch_size
        )

        train_sample_count += current_batch_size


        if batch_index % 50 == 0:
            print(
                f"train batch "
                f"{batch_index}/{len(train_bleed_dl)} "
                f"loss {loss_train_value.item():.6f}"
            )


    avg_train_loss = (
        train_loss_sum
        / train_sample_count
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_sum = 0.0
    val_sample_count = 0

    total_tp = None
    total_fp = None
    total_fn = None
    total_tn = None


    with torch.no_grad():

        for batch_index, (
            val_img,
            val_mask,
        ) in enumerate(valid_bleed_dl):

            val_img = val_img.to(
                "cuda",
                non_blocking=True,
            )

            val_mask = prepare_mask(
                val_mask
            )

            logits = model(
                val_img
            )

            loss_valid_value = loss_function(
                logits,
                val_mask,
            )


            current_batch_size = val_img.size(0)

            val_loss_sum += (
                loss_valid_value.item()
                * current_batch_size
            )

            val_sample_count += current_batch_size


            predictions = get_predictions(
                logits
            )


            batch_tp, batch_fp, batch_fn, batch_tn = (
                get_segmentation_stats(
                    predictions,
                    val_mask,
                )
            )


            batch_tp = batch_tp.sum(
                dim=0
            )

            batch_fp = batch_fp.sum(
                dim=0
            )

            batch_fn = batch_fn.sum(
                dim=0
            )

            batch_tn = batch_tn.sum(
                dim=0
            )


            if total_tp is None:

                total_tp = batch_tp
                total_fp = batch_fp
                total_fn = batch_fn
                total_tn = batch_tn

            else:

                total_tp += batch_tp
                total_fp += batch_fp
                total_fn += batch_fn
                total_tn += batch_tn


            if batch_index % 50 == 0:
                print(
                    f"validation batch "
                    f"{batch_index}/{len(valid_bleed_dl)} "
                    f"loss {loss_valid_value.item():.6f}"
                )


    avg_val_loss = (
        val_loss_sum
        / val_sample_count
    )


    iou_classes = smp.metrics.iou_score(
        total_tp,
        total_fp,
        total_fn,
        total_tn,
        reduction="none",
    )


    dice_classes = smp.metrics.f1_score(
        total_tp,
        total_fp,
        total_fn,
        total_tn,
        reduction="none",
    )


    blood_iou = iou_classes[
        BLOOD_CLASS_INDEX
    ].item()


    blood_dice = dice_classes[
        BLOOD_CLASS_INDEX
    ].item()


    # Save the model with the highest validation Dice for blood.
    if blood_dice > best_val_dice:

        best_val_dice = blood_dice
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

        print(
            "New best checkpoint saved."
        )

        print(
            f"Best validation Dice: "
            f"{best_val_dice:.6f}"
        )

        print(
            f"Best epoch: {best_epoch}"
        )


    scheduler.step()

    current_learning_rate = (
        scheduler.get_last_lr()[0]
    )


    print(
        f"\n----- Average values for epoch "
        f"{epoch + 1} -----"
    )

    print(
        f"avg_train_loss: {avg_train_loss:.6f}"
    )

    print(
        f"avg_val_loss: {avg_val_loss:.6f}"
    )

    print(
        f"blood_iou: {blood_iou:.6f}"
    )

    print(
        f"blood_dice: {blood_dice:.6f}"
    )

    print(
        f"learning_rate: {current_learning_rate}"
    )

    print(
        f"best_epoch: {best_epoch}"
    )

    print(
        f"best_validation_dice: "
        f"{best_val_dice:.6f}"
    )


print(
    "\n======== RABBANI TRAINING COMPLETED ========"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation Dice: {best_val_dice:.6f}"
)

print(
    f"Checkpoint saved to: {checkpoint_path}"
)
