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

    CrossEntropyLoss is used as the only segmentation loss.

    The training procedure includes several measures intended to improve
    optimization stability:

    - deterministic random seeds
    - batch size 2 with gradient accumulation
    - removal of incomplete training batches
    - frozen encoder BatchNorm running statistics
    - lower learning rate for the pretrained encoder
    - AdamW optimizer
    - gradient clipping
    - adaptive learning-rate reduction
    - early stopping
    - checkpoint selection using both global and per-image Dice

usage:
    python -m scripts.rabbani.train_rabbani 60

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
    create_bleed_train_transform,
    create_bleed_eval_transform,
)
from src.hemoset_dataset_v2 import CustomImageDataset


# ============================================================
# FIXED MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "unet"
ENCODER_NAME = "resnet18"

SEGMENTATION_MODE = "multiclass"
NUM_CLASSES = 2

BACKGROUND_CLASS_INDEX = 0
BLOOD_CLASS_INDEX = 1

DEVICE = torch.device("cuda")


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# A physical batch size of 2 is safer for 480x864 images on the Tesla K80.
BATCH_SIZE = 2

# Two consecutive batches are accumulated before updating the weights.
# The effective batch size is therefore approximately 4.
ACCUMULATION_STEPS = 2

NUM_WORKERS = 2

# Use a lower learning rate for the pretrained encoder and a higher learning
# rate for the randomly initialized decoder and segmentation head.
ENCODER_LEARNING_RATE = 1e-4
DECODER_LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

# Prevent unusually difficult batches from producing excessively large
# parameter updates.
MAX_GRAD_NORM = 1.0

# Reduce learning rates when the validation selection score stops improving.
LR_REDUCTION_FACTOR = 0.5
LR_PATIENCE = 4
LR_THRESHOLD = 0.001
MIN_LEARNING_RATE = 1e-6

# Stop training after this number of epochs without a meaningful improvement.
EARLY_STOPPING_PATIENCE = 12

# A new checkpoint is saved only when the score improves by at least this
# amount.
MIN_CHECKPOINT_IMPROVEMENT = 0.001

# The checkpoint selection score considers both pixel-level performance and
# consistency across individual images.
GLOBAL_DICE_WEIGHT = 0.5
MEAN_IMAGE_DICE_WEIGHT = 0.5

RANDOM_SEED = 42


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings on the CPU node.
torch.backends.nnpack.set_flags(False)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def configure_reproducibility(seed):
    """
    Configure the random number generators used by Python, NumPy and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def seed_worker(worker_id):
    """
    Give each DataLoader worker a deterministic random seed.
    """
    worker_seed = torch.initial_seed() % (2**32)

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
# HELPER FUNCTIONS
# ============================================================

def prepare_mask(mask):
    """
    Convert the mask from [B, 1, H, W] to [B, H, W] and prepare it for
    CrossEntropyLoss.
    """
    return torch.squeeze(
        mask,
        dim=1,
    ).long().to(
        DEVICE,
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
    Compute TP, FP, FN and TN separately for every image and class.
    """
    return smp.metrics.get_stats(
        predictions,
        mask,
        mode=SEGMENTATION_MODE,
        num_classes=NUM_CLASSES,
    )


def freeze_encoder_batch_norm_statistics(model):
    """
    Keep the running mean and variance of the pretrained encoder BatchNorm
    layers fixed.

    BatchNorm affine parameters remain trainable because requires_grad is not
    modified. Only the running statistics are prevented from being updated
    using very small batches.
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


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured for GPU training."
    )


# Read the maximum number of epochs from the command line.
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

    # Avoid a final batch containing only one image. This is especially
    # important for the trainable BatchNorm layers in the U-Net decoder.
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


# Inspect the shape of the first training batch.
train_img, train_mask = next(
    iter(train_bleed_dl)
)


print(
    "======== RABBANI STABLE TRAINING CONFIGURATION ========"
)

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
    f"Training batches: {len(train_bleed_dl)}"
)

print(
    f"Validation batches: {len(valid_bleed_dl)}"
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
    f"Maximum epochs: {n_epochs}"
)

print(
    f"Physical batch size: {BATCH_SIZE}"
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
    f"Random seed: {RANDOM_SEED}"
)


# ============================================================
# MODEL
# ============================================================

model = smp.Unet(
    encoder_name=ENCODER_NAME,
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
).to(
    DEVICE
)


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
# LOSS
# ============================================================

# Keep CrossEntropyLoss as the only segmentation loss.
loss_function = torch.nn.CrossEntropyLoss().to(
    DEVICE
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


best_selection_score = float("-inf")
best_global_dice = float("-inf")
best_mean_image_dice = float("-inf")

best_epoch = 0
epochs_without_improvement = 0


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

    # model.train() puts every BatchNorm layer into training mode.
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


        loss_train_value = loss_function(
            logits,
            train_mask,
        )


        if not torch.isfinite(
            loss_train_value
        ):
            raise FloatingPointError(
                "Non-finite training loss detected at "
                f"epoch {epoch + 1}, batch {batch_index}."
            )


        # Determine the number of batches in the current accumulation group.
        # This also correctly handles a possible final incomplete group.
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


            loss_valid_value = loss_function(
                logits,
                val_mask,
            )


            if not torch.isfinite(
                loss_valid_value
            ):
                raise FloatingPointError(
                    "Non-finite validation loss detected at "
                    f"epoch {epoch + 1}, batch {batch_index}."
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


            batch_tp, batch_fp, batch_fn, batch_tn = (
                get_segmentation_stats(
                    predictions,
                    val_mask,
                )
            )


            # Keep one row for every image. This allows both global and
            # per-image metrics to be calculated.
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


    # Join validation statistics while preserving the image dimension.
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


    # Combine global pixel-level quality and image-level consistency.
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
        > best_selection_score
        + MIN_CHECKPOINT_IMPROVEMENT
    )


    if meaningful_improvement:

        best_selection_score = selection_score
        best_global_dice = global_blood_dice
        best_mean_image_dice = mean_image_dice

        best_epoch = epoch + 1
        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            checkpoint_path,
        )


        print(
            "New best stable checkpoint saved."
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
            f"Best epoch: {best_epoch}"
        )

    else:

        epochs_without_improvement += 1


    # Reduce the learning rate only when validation performance plateaus.
    scheduler.step(
        selection_score
    )


    current_learning_rates = (
        get_optimizer_learning_rates(
            optimizer
        )
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
        f"avg_gradient_norm: {avg_gradient_norm:.6f}"
    )

    print(
        f"global_blood_iou: {global_blood_iou:.6f}"
    )

    print(
        f"global_blood_dice: {global_blood_dice:.6f}"
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
        f"selection_score: {selection_score:.6f}"
    )

    print(
        "learning_rates: "
        f"{current_learning_rates}"
    )

    print(
        f"best_epoch: {best_epoch}"
    )

    print(
        f"best_selection_score: "
        f"{best_selection_score:.6f}"
    )

    print(
        "epochs_without_improvement: "
        f"{epochs_without_improvement}"
    )


    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping activated."
        )

        print(
            f"No meaningful validation improvement for "
            f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
        )

        break


print(
    "\n======== RABBANI TRAINING COMPLETED ========"
)

print(
    f"Best epoch: {best_epoch}"
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
    f"Checkpoint saved to: {checkpoint_path}"
)