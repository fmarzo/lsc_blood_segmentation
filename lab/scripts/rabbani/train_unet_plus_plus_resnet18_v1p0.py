"""
file: train_unet_plus_plus_resnet18_v1p0.py

brief:
    Train a U-Net++ ResNet-18 model on the Rabbani Bleeding Segmentation
    dataset.

    The segmentation approach is selected through:

        config_split.SEGMENTATION_MODE

    Supported values:

        "binary"
        "multiclass"

    Binary segmentation:

    - the model produces one output channel representing blood;
    - BCEWithLogitsLoss evaluates every pixel independently;
    - DiceLoss encourages overlap between predicted and ground-truth blood;
    - the total loss is BCEWithLogitsLoss + DiceLoss;
    - predictions are obtained using sigmoid and BINARY_THRESHOLD.

    Multiclass segmentation:

    - the model produces two output channels;
    - class 0 represents background;
    - class 1 represents blood;
    - CrossEntropyLoss is used;
    - predictions are obtained using argmax.

    The architecture is fixed to U-Net++ with a ResNet-18 encoder pretrained
    on ImageNet.

    The training procedure includes:

    - deterministic random seeds
    - physical batch size 2
    - gradient accumulation over two batches
    - effective batch size approximately 4
    - removal of incomplete training batches
    - frozen encoder BatchNorm running statistics
    - separate encoder and decoder learning rates
    - AdamW optimizer
    - gradient clipping
    - adaptive learning-rate reduction
    - early stopping
    - checkpoint selection using global and per-image Dice

usage:
    python -m scripts.rabbani.train_unet_plus_plus_resnet18_v1p0 60

    If the number of epochs is omitted, config_split.DEFAULT_EPOCHS is used.
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
    create_bleed_eval_transform,
    create_bleed_train_transform,
)
from src.hemoset_dataset_v2 import CustomImageDataset


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "unet_plus_plus"
ENCODER_NAME = "resnet18"

SEGMENTATION_MODE = (
    config_split.SEGMENTATION_MODE
    .strip()
    .lower()
)

SUPPORTED_SEGMENTATION_MODES = {
    "binary",
    "multiclass",
}


if SEGMENTATION_MODE not in SUPPORTED_SEGMENTATION_MODES:
    raise ValueError(
        "Unsupported SEGMENTATION_MODE value: "
        f"{SEGMENTATION_MODE}. "
        "Supported values are 'binary' and 'multiclass'."
    )


if SEGMENTATION_MODE == "binary":

    NUM_OUTPUT_CHANNELS = 1
    BLOOD_CLASS_INDEX = 0

    BINARY_THRESHOLD = getattr(
        config_split,
        "BINARY_THRESHOLD",
        0.50,
    )

else:

    NUM_OUTPUT_CHANNELS = 2

    BACKGROUND_CLASS_INDEX = 0
    BLOOD_CLASS_INDEX = 1

    BINARY_THRESHOLD = None


DEVICE = torch.device("cuda")


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Rabbani images use a 480 x 864 resolution. A physical batch size of two is
# safer on the Tesla K80.
BATCH_SIZE = 2

# Two consecutive physical batches are accumulated before updating the model.
ACCUMULATION_STEPS = 2

NUM_WORKERS = 2

ENCODER_LEARNING_RATE = 1e-4
DECODER_LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

MAX_GRAD_NORM = 1.0

EARLY_STOPPING_PATIENCE = 12
MIN_CHECKPOINT_IMPROVEMENT = 0.001

GLOBAL_DICE_WEIGHT = 0.5
MEAN_IMAGE_DICE_WEIGHT = 0.5

RANDOM_SEED = 42


# ============================================================
# SCHEDULER CONFIGURATION
# ============================================================

LR_REDUCTION_FACTOR = 0.5
LR_PATIENCE = 4
LR_THRESHOLD = 0.001
MIN_LEARNING_RATE = 1e-6


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured for GPU training."
    )


# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings.
torch.backends.nnpack.set_flags(False)

torch.backends.cudnn.benchmark = False


# ============================================================
# REPRODUCIBILITY
# ============================================================

def configure_reproducibility(seed):
    """
    Seed Python, NumPy and PyTorch.

    Exact deterministic CUDA execution is not forced because some
    segmentation operations do not provide deterministic implementations.
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


data_generator = torch.Generator()

data_generator.manual_seed(
    RANDOM_SEED
)


# ============================================================
# SEGMENTATION HELPERS
# ============================================================

def prepare_mask(mask):
    """
    Prepare masks according to the selected segmentation mode.

    Binary masks keep the [B, 1, H, W] shape and use floating-point values.

    Multiclass masks are converted from [B, 1, H, W] to [B, H, W] and use
    integer class indices.
    """
    if SEGMENTATION_MODE == "binary":

        return mask.float().to(
            DEVICE,
            non_blocking=True,
        )

    return torch.squeeze(
        mask,
        dim=1,
    ).long().to(
        DEVICE,
        non_blocking=True,
    )


def compute_loss(
    logits,
    mask,
    bce_loss,
    dice_loss,
    cross_entropy_loss,
):
    """
    Compute the loss corresponding to the selected segmentation mode.
    """
    if SEGMENTATION_MODE == "binary":

        bce_value = bce_loss(
            logits,
            mask,
        )

        dice_value = dice_loss(
            logits,
            mask,
        )

        return (
            bce_value
            + dice_value
        )

    return cross_entropy_loss(
        logits,
        mask,
    )


def get_predictions(logits):
    """
    Convert model logits into the final predicted segmentation map.
    """
    if SEGMENTATION_MODE == "binary":

        probabilities = torch.sigmoid(
            logits
        )

        return (
            probabilities
            >= BINARY_THRESHOLD
        ).long()

    return torch.argmax(
        logits,
        dim=1,
    )


def get_segmentation_stats(
    predictions,
    mask,
):
    """
    Compute TP, FP, FN and TN separately for every image.
    """
    if SEGMENTATION_MODE == "binary":

        return smp.metrics.get_stats(
            predictions,
            mask.long(),
            mode="binary",
        )

    return smp.metrics.get_stats(
        predictions,
        mask,
        mode="multiclass",
        num_classes=NUM_OUTPUT_CHANNELS,
    )


def validate_output_shape(
    logits,
    mask,
):
    """
    Verify that output and target dimensions match the segmentation mode.
    """
    if SEGMENTATION_MODE == "binary":

        expected_shape = tuple(
            mask.shape
        )

    else:

        expected_shape = (
            mask.shape[0],
            NUM_OUTPUT_CHANNELS,
            mask.shape[1],
            mask.shape[2],
        )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected model output shape. "
            f"Received {tuple(logits.shape)}, "
            f"expected {expected_shape} for "
            f"{SEGMENTATION_MODE} segmentation."
        )


def freeze_encoder_batch_norm_statistics(model):
    """
    Keep the running statistics of pretrained encoder BatchNorm layers fixed.

    BatchNorm affine parameters remain trainable.
    """
    for module in model.encoder.modules():

        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm,
        ):
            module.eval()


def get_optimizer_learning_rates(optimizer):
    """
    Return the current learning rate of every optimizer parameter group.
    """
    return [
        parameter_group["lr"]
        for parameter_group in optimizer.param_groups
    ]


# ============================================================
# NUMBER OF EPOCHS
# ============================================================

if len(sys.argv) > 1:

    n_epochs = int(
        sys.argv[1]
    )

else:

    n_epochs = config_split.DEFAULT_EPOCHS


if n_epochs <= 0:
    raise ValueError(
        "The number of epochs must be greater than zero."
    )


# ============================================================
# RABBANI DATASET AND TRANSFORMS
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

train_bleed_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=True,

    # Avoid a final batch containing only one image.
    drop_last=True,

    worker_init_fn=seed_worker,
    generator=data_generator,

    persistent_workers=(
        NUM_WORKERS > 0
    ),
)


valid_bleed_dl = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
    drop_last=False,

    worker_init_fn=seed_worker,

    persistent_workers=(
        NUM_WORKERS > 0
    ),
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
# FIRST BATCH INSPECTION
# ============================================================

train_img, train_mask = next(
    iter(train_bleed_dl)
)


print(
    "======== RABBANI U-NET++ TRAINING CONFIGURATION ========"
)

print(
    f"Feature batch shape: "
    f"{train_img.size()}"
)

print(
    f"Labels batch shape: "
    f"{train_mask.size()}"
)

print(
    f"Training CSV: "
    f"{config_split.CSV_TRAIN_PATH_V1P0}"
)

print(
    f"Validation CSV: "
    f"{config_split.CSV_VALID_PATH_V1P0}"
)

print(
    f"Training samples: "
    f"{len(train_ds)}"
)

print(
    f"Validation samples: "
    f"{len(valid_ds)}"
)

print(
    f"Training batches: "
    f"{len(train_bleed_dl)}"
)

print(
    f"Validation batches: "
    f"{len(valid_bleed_dl)}"
)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Encoder: {ENCODER_NAME}"
)

print(
    f"Segmentation mode: "
    f"{SEGMENTATION_MODE}"
)

print(
    f"Output channels: "
    f"{NUM_OUTPUT_CHANNELS}"
)

if SEGMENTATION_MODE == "binary":

    print(
        "Training loss: BCEWithLogitsLoss + DiceLoss"
    )

    print(
        f"Binary threshold: "
        f"{BINARY_THRESHOLD:.2f}"
    )

else:

    print(
        "Training loss: CrossEntropyLoss"
    )

print(
    f"Maximum epochs: "
    f"{n_epochs}"
)

print(
    f"Physical batch size: "
    f"{BATCH_SIZE}"
)

print(
    "Effective batch size: "
    f"{BATCH_SIZE * ACCUMULATION_STEPS}"
)

print(
    f"Encoder learning rate: "
    f"{ENCODER_LEARNING_RATE}"
)

print(
    f"Decoder learning rate: "
    f"{DECODER_LEARNING_RATE}"
)

print(
    f"Random seed: "
    f"{RANDOM_SEED}"
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
# OUTPUT SHAPE CHECK
# ============================================================

with torch.inference_mode():

    shape_check_images = train_img.to(
        DEVICE
    )

    shape_check_masks = prepare_mask(
        train_mask
    )

    shape_check_logits = model(
        shape_check_images
    )

    validate_output_shape(
        shape_check_logits,
        shape_check_masks,
    )


print(
    f"Model output shape: "
    f"{tuple(shape_check_logits.shape)}"
)


del shape_check_images
del shape_check_masks
del shape_check_logits
del train_img
del train_mask


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.encoder.parameters(),
            "lr": ENCODER_LEARNING_RATE,
        },
        {
            "params": model.decoder.parameters(),
            "lr": DECODER_LEARNING_RATE,
        },
        {
            "params": model.segmentation_head.parameters(),
            "lr": DECODER_LEARNING_RATE,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=LR_REDUCTION_FACTOR,
    patience=LR_PATIENCE,
    threshold=LR_THRESHOLD,
    threshold_mode="abs",
    cooldown=1,
    min_lr=MIN_LEARNING_RATE,
)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

if SEGMENTATION_MODE == "binary":

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

    cross_entropy_loss = None

else:

    bce_loss = None
    dice_loss = None

    cross_entropy_loss = (
        torch.nn.CrossEntropyLoss()
        .to(DEVICE)
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
        f"unet_plus_plus_{SEGMENTATION_MODE}_"
        "best_dice_bleed_seg_"
        f"{ENCODER_NAME}.pth"
    ),
)


best_selection_score = float("-inf")

best_global_dice = float("-inf")
best_mean_image_dice = float("-inf")

best_epoch = 0

epochs_without_improvement = 0


print(
    f"Checkpoint: "
    f"{checkpoint_path}"
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(n_epochs):

    print(
        f"\n------------ EPOCH: "
        f"{epoch + 1}/{n_epochs} "
        f"------------"
    )

    model.train()

    # model.train() changes every BatchNorm layer to training mode.
    # Restore only the pretrained encoder BatchNorm layers to evaluation mode.
    freeze_encoder_batch_norm_statistics(
        model
    )


    train_loss_sum = 0.0
    train_sample_count = 0

    gradient_norm_sum = 0.0
    optimizer_step_count = 0


    optimizer.zero_grad(
        set_to_none=True
    )


    number_of_train_batches = len(
        train_bleed_dl
    )


    for batch_index, (
        train_img,
        train_mask,
    ) in enumerate(train_bleed_dl):

        train_img = train_img.to(
            DEVICE,
            non_blocking=True,
        )

        train_mask = prepare_mask(
            train_mask
        )


        logits = model(
            train_img
        )


        if batch_index == 0:
            validate_output_shape(
                logits,
                train_mask,
            )


        loss_train_value = compute_loss(
            logits=logits,
            mask=train_mask,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
            cross_entropy_loss=cross_entropy_loss,
        )


        if not torch.isfinite(
            loss_train_value
        ):
            raise FloatingPointError(
                "Non-finite training loss detected at "
                f"epoch {epoch + 1}, "
                f"batch {batch_index}."
            )


        accumulation_group_start = (
            batch_index
            // ACCUMULATION_STEPS
            * ACCUMULATION_STEPS
        )


        accumulation_group_end = min(
            accumulation_group_start
            + ACCUMULATION_STEPS,
            number_of_train_batches,
        )


        current_accumulation_group_size = (
            accumulation_group_end
            - accumulation_group_start
        )


        scaled_loss = (
            loss_train_value
            / current_accumulation_group_size
        )


        scaled_loss.backward()


        should_update_weights = (
            batch_index + 1
            == accumulation_group_end
        )


        if should_update_weights:

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=MAX_GRAD_NORM,
                )
            )


            optimizer.step()


            optimizer.zero_grad(
                set_to_none=True
            )


            gradient_norm_sum += float(
                gradient_norm.detach().cpu()
            )

            optimizer_step_count += 1


        current_batch_size = train_img.size(
            0
        )


        train_loss_sum += (
            loss_train_value.item()
            * current_batch_size
        )


        train_sample_count += (
            current_batch_size
        )


        if batch_index % 50 == 0:

            print(
                f"train batch "
                f"{batch_index}/{number_of_train_batches} "
                f"loss {loss_train_value.item():.6f}"
            )


    avg_train_loss = (
        train_loss_sum
        / train_sample_count
    )


    if optimizer_step_count > 0:

        avg_gradient_norm = (
            gradient_norm_sum
            / optimizer_step_count
        )

    else:

        avg_gradient_norm = 0.0


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_loss_sum = 0.0
    val_sample_count = 0


    tp_batches = []
    fp_batches = []
    fn_batches = []
    tn_batches = []


    with torch.inference_mode():

        for batch_index, (
            val_img,
            val_mask,
        ) in enumerate(valid_bleed_dl):

            val_img = val_img.to(
                DEVICE,
                non_blocking=True,
            )

            val_mask = prepare_mask(
                val_mask
            )


            logits = model(
                val_img
            )


            if batch_index == 0:
                validate_output_shape(
                    logits,
                    val_mask,
                )


            loss_valid_value = compute_loss(
                logits=logits,
                mask=val_mask,
                bce_loss=bce_loss,
                dice_loss=dice_loss,
                cross_entropy_loss=cross_entropy_loss,
            )


            if not torch.isfinite(
                loss_valid_value
            ):
                raise FloatingPointError(
                    "Non-finite validation loss detected at "
                    f"epoch {epoch + 1}, "
                    f"batch {batch_index}."
                )


            current_batch_size = val_img.size(
                0
            )


            val_loss_sum += (
                loss_valid_value.item()
                * current_batch_size
            )


            val_sample_count += (
                current_batch_size
            )


            predictions = get_predictions(
                logits
            )


            (
                batch_tp,
                batch_fp,
                batch_fn,
                batch_tn,
            ) = get_segmentation_stats(
                predictions,
                val_mask,
            )


            # Preserve one row for every validation image.
            tp_batches.append(
                batch_tp.cpu()
            )

            fp_batches.append(
                batch_fp.cpu()
            )

            fn_batches.append(
                batch_fn.cpu()
            )

            tn_batches.append(
                batch_tn.cpu()
            )


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


    val_tp = torch.cat(
        tp_batches,
        dim=0,
    )

    val_fp = torch.cat(
        fp_batches,
        dim=0,
    )

    val_fn = torch.cat(
        fn_batches,
        dim=0,
    )

    val_tn = torch.cat(
        tn_batches,
        dim=0,
    )


    # ========================================================
    # PER-IMAGE VALIDATION METRICS
    # ========================================================

    per_image_iou_classes = smp.metrics.iou_score(
        val_tp,
        val_fp,
        val_fn,
        val_tn,
        reduction="none",
    )


    per_image_dice_classes = smp.metrics.f1_score(
        val_tp,
        val_fp,
        val_fn,
        val_tn,
        reduction="none",
    )


    blood_iou_per_image = per_image_iou_classes[
        :,
        BLOOD_CLASS_INDEX,
    ]


    blood_dice_per_image = per_image_dice_classes[
        :,
        BLOOD_CLASS_INDEX,
    ]


    mean_image_iou = (
        blood_iou_per_image
        .mean()
        .item()
    )


    std_image_iou = (
        blood_iou_per_image
        .std(correction=0)
        .item()
    )


    mean_image_dice = (
        blood_dice_per_image
        .mean()
        .item()
    )


    std_image_dice = (
        blood_dice_per_image
        .std(correction=0)
        .item()
    )


    # ========================================================
    # GLOBAL VALIDATION METRICS
    # ========================================================

    global_tp = val_tp.sum(
        dim=0
    )

    global_fp = val_fp.sum(
        dim=0
    )

    global_fn = val_fn.sum(
        dim=0
    )

    global_tn = val_tn.sum(
        dim=0
    )


    global_iou_classes = smp.metrics.iou_score(
        global_tp,
        global_fp,
        global_fn,
        global_tn,
        reduction="none",
    )


    global_dice_classes = smp.metrics.f1_score(
        global_tp,
        global_fp,
        global_fn,
        global_tn,
        reduction="none",
    )


    global_precision_classes = smp.metrics.precision(
        global_tp,
        global_fp,
        global_fn,
        global_tn,
        reduction="none",
    )


    global_recall_classes = smp.metrics.recall(
        global_tp,
        global_fp,
        global_fn,
        global_tn,
        reduction="none",
    )


    global_blood_iou = global_iou_classes[
        BLOOD_CLASS_INDEX
    ].item()


    global_blood_dice = global_dice_classes[
        BLOOD_CLASS_INDEX
    ].item()


    global_blood_precision = global_precision_classes[
        BLOOD_CLASS_INDEX
    ].item()


    global_blood_recall = global_recall_classes[
        BLOOD_CLASS_INDEX
    ].item()


    selection_score = (
        GLOBAL_DICE_WEIGHT
        * global_blood_dice
        +
        MEAN_IMAGE_DICE_WEIGHT
        * mean_image_dice
    )


    if not np.isfinite(
        selection_score
    ):
        raise FloatingPointError(
            "Non-finite validation selection score detected."
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

        best_global_dice = (
            global_blood_dice
        )

        best_mean_image_dice = (
            mean_image_dice
        )

        best_epoch = epoch + 1

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            checkpoint_path,
        )


        print(
            "\nNew best checkpoint saved."
        )

        print(
            f"Best selection score: "
            f"{best_selection_score:.6f}"
        )

        print(
            f"Best global Dice: "
            f"{best_global_dice:.6f}"
        )

        print(
            f"Best mean-image Dice: "
            f"{best_mean_image_dice:.6f}"
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

    else:

        epochs_without_improvement += 1


    scheduler.step(
        selection_score
    )


    current_learning_rates = (
        get_optimizer_learning_rates(
            optimizer
        )
    )


    # ========================================================
    # EPOCH SUMMARY
    # ========================================================

    print(
        f"\n----- Average values for epoch "
        f"{epoch + 1} -----"
    )

    print(
        f"avg_train_loss: "
        f"{avg_train_loss:.6f}"
    )

    print(
        f"avg_val_loss: "
        f"{avg_val_loss:.6f}"
    )

    print(
        f"avg_gradient_norm: "
        f"{avg_gradient_norm:.6f}"
    )

    print(
        f"global_blood_iou: "
        f"{global_blood_iou:.6f}"
    )

    print(
        f"global_blood_dice: "
        f"{global_blood_dice:.6f}"
    )

    print(
        f"global_blood_precision: "
        f"{global_blood_precision:.6f}"
    )

    print(
        f"global_blood_recall: "
        f"{global_blood_recall:.6f}"
    )

    print(
        f"mean_image_iou: "
        f"{mean_image_iou:.6f} "
        f"+/- {std_image_iou:.6f}"
    )

    print(
        f"mean_image_dice: "
        f"{mean_image_dice:.6f} "
        f"+/- {std_image_dice:.6f}"
    )

    print(
        f"selection_score: "
        f"{selection_score:.6f}"
    )

    print(
        "learning_rates: "
        f"{current_learning_rates}"
    )

    print(
        f"best_epoch: "
        f"{best_epoch}"
    )

    print(
        f"best_selection_score: "
        f"{best_selection_score:.6f}"
    )

    print(
        "epochs_without_improvement: "
        f"{epochs_without_improvement}"
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

        print(
            "No meaningful validation improvement for "
            f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
        )

        break


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n======== RABBANI TRAINING COMPLETED ========"
)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Encoder: {ENCODER_NAME}"
)

print(
    f"Segmentation mode: "
    f"{SEGMENTATION_MODE}"
)

print(
    f"Output channels: "
    f"{NUM_OUTPUT_CHANNELS}"
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Best selection score: "
    f"{best_selection_score:.6f}"
)

print(
    f"Best global validation Dice: "
    f"{best_global_dice:.6f}"
)

print(
    f"Best mean-image validation Dice: "
    f"{best_mean_image_dice:.6f}"
)

print(
    f"Checkpoint saved to: "
    f"{checkpoint_path}"
)