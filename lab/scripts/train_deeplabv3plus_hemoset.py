"""
file: train_deeplabv3plus_hemoset.py

brief:
    Train a DeepLabV3+ segmentation model on HemoSet.

    Fixed model configuration:

    - architecture: DeepLabV3+
    - encoder: ResNet-18 pretrained on ImageNet
    - encoder output stride: 16
    - segmentation mode: multiclass
    - classes: background and blood
    - loss: CrossEntropyLoss
    - physical batch size: 4

    The training split receives HemoSet online augmentation, while the
    validation split receives only the HemoSet evaluation preprocessing.

    The checkpoint is selected using a balanced score composed of:

        0.5 * dataset-level Dice
        0.5 * mean per-image Dice

usage:
    python -m scripts.train_deeplabv3plus_hemoset 50
"""

import os
import random
import sys

import numpy as np
import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import (
    create_eval_transform,
    create_train_transform,
)
from src.hemoset_dataset_v2 import CustomImageDataset


# ============================================================
# FIXED MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "deeplabv3plus"
ENCODER_NAME = "resnet18"

SEGMENTATION_MODE = "multiclass"
NUM_CLASSES = 2

BACKGROUND_CLASS_INDEX = 0
BLOOD_CLASS_INDEX = 1

ENCODER_OUTPUT_STRIDE = 16
DECODER_CHANNELS = 256
DECODER_ATROUS_RATES = (12, 24, 36)
UPSAMPLING_FACTOR = 4


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 4
NUM_WORKERS = 2

ENCODER_LEARNING_RATE = 1e-4
DECODER_LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_MAX_NORM = 1.0

RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 12
MIN_CHECKPOINT_IMPROVEMENT = 0.001

GLOBAL_DICE_WEIGHT = 0.5
MEAN_IMAGE_DICE_WEIGHT = 0.5

LR_REDUCTION_FACTOR = 0.5
LR_PATIENCE = 4
LR_THRESHOLD = 0.001
MINIMUM_LEARNING_RATE = 1e-6

EPSILON = 1e-12


# ============================================================
# DEVICE AND HARDWARE CONFIGURATION
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This training script is configured for GPU execution."
    )


DEVICE = torch.device("cuda")


# The installed cuDNN version is not compatible with the Tesla K80.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings.
torch.backends.nnpack.set_flags(False)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def configure_reproducibility(seed):
    """
    Seed Python, NumPy and PyTorch.

    Exact deterministic CUDA execution is not enabled because some
    segmentation operations do not provide deterministic implementations.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """
    Seed every DataLoader worker.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ============================================================
# MODEL HELPERS
# ============================================================

def create_model():
    """
    Create the DeepLabV3+ ResNet-18 model.
    """
    return smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet",
        encoder_output_stride=ENCODER_OUTPUT_STRIDE,
        decoder_channels=DECODER_CHANNELS,
        decoder_atrous_rates=DECODER_ATROUS_RATES,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
        upsampling=UPSAMPLING_FACTOR,
    )


def freeze_encoder_batch_norm_statistics(model):
    """
    Freeze the running statistics of the encoder BatchNorm layers.

    The affine parameters remain trainable.
    """
    for module in model.encoder.modules():
        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm,
        ):
            module.eval()


def prepare_mask(mask):
    """
    Convert masks from [B, 1, H, W] to [B, H, W].
    """
    return torch.squeeze(
        mask,
        dim=1,
    ).long().to(
        DEVICE,
        non_blocking=True,
    )


def validate_output_shape(logits, masks):
    """
    Verify that model output and masks have compatible spatial dimensions.
    """
    expected_shape = (
        masks.shape[0],
        NUM_CLASSES,
        masks.shape[1],
        masks.shape[2],
    )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected DeepLabV3+ output shape. "
            f"Received {tuple(logits.shape)}, "
            f"expected {expected_shape}."
        )


# ============================================================
# NUMBER OF EPOCHS
# ============================================================

if len(sys.argv) > 1:
    n_epochs = int(sys.argv[1])
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


configure_reproducibility(
    RANDOM_SEED
)


# ============================================================
# HEMOSET TRANSFORMS
# ============================================================

train_transform = create_train_transform()

eval_transform = create_eval_transform()


# ============================================================
# HEMOSET DATASETS
# ============================================================

train_ds = CustomImageDataset(
    config_split.CSV_TRAIN_PATH,
    train_transform,
)


valid_ds = CustomImageDataset(
    config_split.CSV_VALID_PATH,
    eval_transform,
)


if len(train_ds) == 0:
    raise ValueError(
        "The HemoSet training dataset is empty."
    )


if len(valid_ds) == 0:
    raise ValueError(
        "The HemoSet validation dataset is empty."
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


train_hemo_DL = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=True,

    # Avoid a final training batch smaller than the physical batch size.
    drop_last=True,

    persistent_workers=(
        NUM_WORKERS > 0
    ),
    worker_init_fn=seed_worker,
    generator=train_generator,
)


valid_hemo_DL = DataLoader(
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


# ============================================================
# DATA SHAPE CHECK
# ============================================================

feature_batch, label_batch = next(
    iter(train_hemo_DL)
)


print(
    f"Feature batch shape: {feature_batch.size()}"
)

print(
    f"Labels batch shape: {label_batch.size()}"
)


# ============================================================
# MODEL
# ============================================================

deeplabv3plus = create_model()

deeplabv3plus.to(
    DEVICE
)


print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Encoder: {ENCODER_NAME}"
)


# Test one batch before starting the complete training.
with torch.inference_mode():

    feature_batch = feature_batch.to(
        DEVICE
    )

    label_batch = prepare_mask(
        label_batch
    )

    output_batch = deeplabv3plus(
        feature_batch
    )

    validate_output_shape(
        output_batch,
        label_batch,
    )


print(
    f"Model output shape: {output_batch.shape}"
)


del feature_batch
del label_batch
del output_batch


# ============================================================
# LOSS
# ============================================================

loss_function = torch.nn.CrossEntropyLoss().to(
    DEVICE
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params": deeplabv3plus.encoder.parameters(),
            "lr": ENCODER_LEARNING_RATE,
        },
        {
            "params": deeplabv3plus.decoder.parameters(),
            "lr": DECODER_LEARNING_RATE,
        },
        {
            "params": deeplabv3plus.segmentation_head.parameters(),
            "lr": DECODER_LEARNING_RATE,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LEARNING-RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=LR_REDUCTION_FACTOR,
    patience=LR_PATIENCE,
    threshold=LR_THRESHOLD,
    threshold_mode="abs",
    cooldown=1,
    min_lr=MINIMUM_LEARNING_RATE,
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
        "deeplabv3plus_multiclass_"
        "best_resnet18_hemo.pth"
    ),
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

print(
    "\n======== HEMOSET DEEPLABV3+ TRAINING CONFIGURATION ========"
)

print(
    f"Training CSV: {config_split.CSV_TRAIN_PATH}"
)

print(
    f"Validation CSV: {config_split.CSV_VALID_PATH}"
)

print(
    f"Training samples: {len(train_ds)}"
)

print(
    f"Validation samples: {len(valid_ds)}"
)

print(
    f"Training batches: {len(train_hemo_DL)}"
)

print(
    f"Validation batches: {len(valid_hemo_DL)}"
)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Encoder: {ENCODER_NAME}"
)

print(
    f"Encoder output stride: {ENCODER_OUTPUT_STRIDE}"
)

print(
    f"Decoder channels: {DECODER_CHANNELS}"
)

print(
    f"Decoder atrous rates: {DECODER_ATROUS_RATES}"
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

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Encoder learning rate: {ENCODER_LEARNING_RATE}"
)

print(
    f"Decoder learning rate: {DECODER_LEARNING_RATE}"
)

print(
    f"Checkpoint: {checkpoint_path}"
)


# ============================================================
# TRAINING STATE
# ============================================================

best_selection_score = float("-inf")

best_epoch = 0

epochs_without_improvement = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(n_epochs):

    print(
        f"\n------------ EPOCH: "
        f"{epoch + 1}/{n_epochs} "
        f"------------"
    )

    # ========================================================
    # TRAIN
    # ========================================================

    deeplabv3plus.train()

    # Keep the pretrained encoder BatchNorm statistics fixed.
    freeze_encoder_batch_norm_statistics(
        deeplabv3plus
    )

    train_loss_sum = 0.0
    train_sample_count = 0

    gradient_norm_sum = 0.0
    optimizer_step_count = 0


    for batch_index, (
        train_images,
        train_masks,
    ) in enumerate(train_hemo_DL):

        optimizer.zero_grad(
            set_to_none=True
        )

        train_images = train_images.to(
            DEVICE,
            non_blocking=True,
        )

        train_masks = prepare_mask(
            train_masks
        )

        logits = deeplabv3plus(
            train_images
        )

        if batch_index == 0:
            validate_output_shape(
                logits,
                train_masks,
            )

        loss_train_value = loss_function(
            logits,
            train_masks,
        )

        if not torch.isfinite(
            loss_train_value
        ):
            raise RuntimeError(
                "Non-finite training loss detected "
                f"at batch {batch_index}: "
                f"{loss_train_value.item()}"
            )

        loss_train_value.backward()


        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                deeplabv3plus.parameters(),
                max_norm=GRADIENT_CLIP_MAX_NORM,
            )
        )


        optimizer.step()


        current_batch_size = train_images.size(
            0
        )

        train_loss_sum += (
            loss_train_value.item()
            * current_batch_size
        )

        train_sample_count += (
            current_batch_size
        )

        gradient_norm_sum += float(
            gradient_norm
        )

        optimizer_step_count += 1


        if batch_index % 50 == 0:
            print(
                f"loss_train "
                f"{loss_train_value.item():.6f}"
            )


    avg_train_loss = (
        train_loss_sum
        / train_sample_count
    )


    avg_gradient_norm = (
        gradient_norm_sum
        / max(optimizer_step_count, 1)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    deeplabv3plus.eval()


    val_loss_sum = 0.0
    val_sample_count = 0
    val_image_count = 0


    global_true_positive = 0.0
    global_false_positive = 0.0
    global_false_negative = 0.0


    per_image_iou_sum = 0.0
    per_image_iou_squared_sum = 0.0

    per_image_dice_sum = 0.0
    per_image_dice_squared_sum = 0.0


    with torch.inference_mode():

        for batch_index, (
            validation_images,
            validation_masks,
        ) in enumerate(valid_hemo_DL):

            validation_images = validation_images.to(
                DEVICE,
                non_blocking=True,
            )

            validation_masks = prepare_mask(
                validation_masks
            )

            logits = deeplabv3plus(
                validation_images
            )

            if batch_index == 0:
                validate_output_shape(
                    logits,
                    validation_masks,
                )

            loss_valid_value = loss_function(
                logits,
                validation_masks,
            )

            if not torch.isfinite(
                loss_valid_value
            ):
                raise RuntimeError(
                    "Non-finite validation loss detected "
                    f"at batch {batch_index}: "
                    f"{loss_valid_value.item()}"
                )


            if batch_index % 50 == 0:
                print(
                    f"loss_valid "
                    f"{loss_valid_value.item():.6f}"
                )


            predictions = torch.argmax(
                logits,
                dim=1,
            )


            predicted_blood = (
                predictions
                == BLOOD_CLASS_INDEX
            )

            target_blood = (
                validation_masks
                == BLOOD_CLASS_INDEX
            )


            true_positive = (
                predicted_blood
                & target_blood
            ).sum(
                dim=(1, 2)
            ).double()


            false_positive = (
                predicted_blood
                & ~target_blood
            ).sum(
                dim=(1, 2)
            ).double()


            false_negative = (
                ~predicted_blood
                & target_blood
            ).sum(
                dim=(1, 2)
            ).double()


            global_true_positive += float(
                true_positive.sum().item()
            )

            global_false_positive += float(
                false_positive.sum().item()
            )

            global_false_negative += float(
                false_negative.sum().item()
            )


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


            per_image_iou_sum += float(
                per_image_iou.sum().item()
            )

            per_image_iou_squared_sum += float(
                (
                    per_image_iou ** 2
                ).sum().item()
            )


            per_image_dice_sum += float(
                per_image_dice.sum().item()
            )

            per_image_dice_squared_sum += float(
                (
                    per_image_dice ** 2
                ).sum().item()
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


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

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


    image_iou_variance = max(
        0.0,
        (
            per_image_iou_squared_sum
            / val_image_count
        )
        - mean_image_iou ** 2,
    )


    image_dice_variance = max(
        0.0,
        (
            per_image_dice_squared_sum
            / val_image_count
        )
        - mean_image_dice ** 2,
    )


    std_image_iou = (
        image_iou_variance ** 0.5
    )


    std_image_dice = (
        image_dice_variance ** 0.5
    )


    selection_score = (
        GLOBAL_DICE_WEIGHT
        * global_dice
        +
        MEAN_IMAGE_DICE_WEIGHT
        * mean_image_dice
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        selection_score
    )


    encoder_current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    decoder_current_lr = (
        optimizer.param_groups[1]["lr"]
    )


    # ========================================================
    # CHECKPOINT SELECTION
    # ========================================================

    meaningful_improvement = (
        selection_score
        >
        best_selection_score
        + MIN_CHECKPOINT_IMPROVEMENT
    )


    if meaningful_improvement:

        best_selection_score = (
            selection_score
        )

        best_epoch = epoch + 1

        epochs_without_improvement = 0


        torch.save(
            deeplabv3plus.state_dict(),
            checkpoint_path,
        )


        print(
            "\nNew best checkpoint saved."
        )


    else:

        epochs_without_improvement += 1


        print(
            "\nNo meaningful validation improvement."
        )

        print(
            "Epochs without improvement: "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )


    # ========================================================
    # EPOCH SUMMARY
    # ========================================================

    print(
        f"\n----- Average values for epoch "
        f"{epoch + 1} -----"
    )

    print(
        f"avg_train_loss {avg_train_loss:.6f}"
    )

    print(
        f"avg_gradient_norm {avg_gradient_norm:.6f}"
    )

    print(
        f"avg_val_loss {avg_val_loss:.6f}"
    )

    print(
        f"global_iou {global_iou:.6f}"
    )

    print(
        f"global_dice {global_dice:.6f}"
    )

    print(
        f"global_precision {global_precision:.6f}"
    )

    print(
        f"global_recall {global_recall:.6f}"
    )

    print(
        f"mean_image_iou "
        f"{mean_image_iou:.6f} "
        f"+/- {std_image_iou:.6f}"
    )

    print(
        f"mean_image_dice "
        f"{mean_image_dice:.6f} "
        f"+/- {std_image_dice:.6f}"
    )

    print(
        f"selection_score {selection_score:.6f}"
    )

    print(
        f"encoder_learning_rate "
        f"{encoder_current_lr:.8f}"
    )

    print(
        f"decoder_learning_rate "
        f"{decoder_current_lr:.8f}"
    )

    print(
        f"best_selection_score "
        f"{best_selection_score:.6f}"
    )

    print(
        f"best_epoch {best_epoch}"
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):
        print(
            "\nEarly stopping activated."
        )

        break


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n======== TRAINING COMPLETED ========"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best selection score: "
    f"{best_selection_score:.6f}"
)

print(
    f"Best checkpoint: {checkpoint_path}"
)