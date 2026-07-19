"""
file: train_deeplabv3plus_rabbani.py

brief:
    Train a DeepLabV3+ ResNet-18 model on the Rabbani Bleeding Segmentation
    dataset.

    The segmentation approach is selected through:

        config_split.SEGMENTATION_MODE

    Supported values:

        "binary"
        "multiclass"

    Binary segmentation:

    - the network produces one output channel representing blood;
    - BCEWithLogitsLoss evaluates each pixel independently;
    - DiceLoss encourages overlap between predicted and ground-truth blood;
    - the total loss is BCEWithLogitsLoss + DiceLoss;
    - predictions use sigmoid followed by config_split.BINARY_THRESHOLD;
    - the blood channel index is 0.

    Multiclass segmentation:

    - the network produces two output channels;
    - class 0 represents background;
    - class 1 represents blood;
    - CrossEntropyLoss is used;
    - predictions are obtained using argmax;
    - the blood class index is 1.

    The architecture is fixed to:

    - DeepLabV3+
    - ResNet-18 encoder pretrained on ImageNet
    - encoder output stride 16
    - decoder channels 256
    - atrous rates 12, 24 and 36
    - physical batch size 4

    The checkpoint name automatically includes the selected segmentation mode.

usage:
    python -m scripts.rabbani.train_deeplabv3plus_rabbani 50
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

MODEL_NAME = "deeplabv3plus"
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


ENCODER_OUTPUT_STRIDE = 16
DECODER_CHANNELS = 256
DECODER_ATROUS_RATES = (12, 24, 36)
UPSAMPLING_FACTOR = 4


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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


# ============================================================
# SCHEDULER CONFIGURATION
# ============================================================

LR_REDUCTION_FACTOR = 0.5
LR_PATIENCE = 4
LR_THRESHOLD = 0.001
MINIMUM_LEARNING_RATE = 1e-6


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

# The installed cuDNN version is not compatible with the Tesla K80.
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

    Exact deterministic CUDA execution is not enabled because some
    segmentation operations do not provide deterministic implementations.
    """
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )


def seed_worker(worker_id):
    """
    Seed every DataLoader worker.
    """
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    random.seed(
        worker_seed
    )

    np.random.seed(
        worker_seed
    )


# ============================================================
# MODEL HELPERS
# ============================================================

def create_model():
    """
    Create the DeepLabV3+ ResNet-18 model.

    The number of output channels depends on SEGMENTATION_MODE.
    """
    return smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet",
        encoder_output_stride=ENCODER_OUTPUT_STRIDE,
        decoder_channels=DECODER_CHANNELS,
        decoder_atrous_rates=DECODER_ATROUS_RATES,
        in_channels=3,
        classes=NUM_OUTPUT_CHANNELS,
        activation=None,
        upsampling=UPSAMPLING_FACTOR,
    )


def freeze_encoder_batch_norm_statistics(model):
    """
    Freeze the running statistics of pretrained encoder BatchNorm layers.

    The affine BatchNorm parameters remain trainable.
    """
    for module in model.encoder.modules():

        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm,
        ):
            module.eval()


# ============================================================
# SEGMENTATION HELPERS
# ============================================================

def prepare_mask(mask):
    """
    Prepare masks according to the selected segmentation mode.

    Binary mode:
        [B, 1, H, W], floating-point values.

    Multiclass mode:
        [B, H, W], integer class indices.
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
    Compute the loss corresponding to SEGMENTATION_MODE.
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
    Convert logits into the final segmentation prediction.
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
    Compute TP, FP, FN and TN independently for every image.
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
    Verify that output and target shapes match the segmentation mode.
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
            "Unexpected DeepLabV3+ output shape. "
            f"Received {tuple(logits.shape)}, "
            f"expected {expected_shape} for "
            f"{SEGMENTATION_MODE} segmentation."
        )


def compute_metrics(
    tp,
    fp,
    fn,
    tn,
):
    """
    Compute IoU, Dice, precision and recall for every image or class.
    """
    iou = smp.metrics.iou_score(
        tp,
        fp,
        fn,
        tn,
        reduction="none",
    )

    dice = smp.metrics.f1_score(
        tp,
        fp,
        fn,
        tn,
        reduction="none",
    )

    precision = smp.metrics.precision(
        tp,
        fp,
        fn,
        tn,
        reduction="none",
    )

    recall = smp.metrics.recall(
        tp,
        fp,
        fn,
        tn,
        reduction="none",
    )

    return (
        iou,
        dice,
        precision,
        recall,
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    bce_loss,
    dice_loss,
    cross_entropy_loss,
):
    """
    Train the model for one epoch.
    """
    model.train()

    # Keep pretrained encoder BatchNorm statistics fixed.
    freeze_encoder_batch_norm_statistics(
        model
    )

    running_loss = 0.0
    number_of_samples = 0

    gradient_norm_sum = 0.0
    optimizer_steps = 0

    for batch_index, (
        train_images,
        train_masks,
    ) in enumerate(train_loader):

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

        logits = model(
            train_images
        )

        if batch_index == 0:
            validate_output_shape(
                logits,
                train_masks,
            )

        loss = compute_loss(
            logits=logits,
            mask=train_masks,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
            cross_entropy_loss=cross_entropy_loss,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite training loss detected "
                f"at batch {batch_index}: "
                f"{loss.item()}"
            )

        loss.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_MAX_NORM,
            )
        )

        optimizer.step()

        current_batch_size = (
            train_images.size(0)
        )

        running_loss += (
            loss.item()
            * current_batch_size
        )

        number_of_samples += (
            current_batch_size
        )

        gradient_norm_sum += float(
            gradient_norm.detach().cpu()
        )

        optimizer_steps += 1

        if batch_index % 50 == 0:

            print(
                f"train batch "
                f"{batch_index}/{len(train_loader)} "
                f"loss {loss.item():.6f}"
            )

    average_loss = (
        running_loss
        / number_of_samples
    )

    average_gradient_norm = (
        gradient_norm_sum
        / max(optimizer_steps, 1)
    )

    return {
        "loss": average_loss,
        "gradient_norm": average_gradient_norm,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_model(
    model,
    validation_loader,
    bce_loss,
    dice_loss,
    cross_entropy_loss,
):
    """
    Evaluate the model on the Rabbani validation split.

    Metrics are computed for the blood channel or class only.
    """
    model.eval()

    running_loss = 0.0
    number_of_samples = 0

    tp_batches = []
    fp_batches = []
    fn_batches = []
    tn_batches = []

    with torch.inference_mode():

        for batch_index, (
            validation_images,
            validation_masks,
        ) in enumerate(validation_loader):

            validation_images = validation_images.to(
                DEVICE,
                non_blocking=True,
            )

            validation_masks = prepare_mask(
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

            loss = compute_loss(
                logits=logits,
                mask=validation_masks,
                bce_loss=bce_loss,
                dice_loss=dice_loss,
                cross_entropy_loss=cross_entropy_loss,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite validation loss detected "
                    f"at batch {batch_index}: "
                    f"{loss.item()}"
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
                validation_masks,
            )

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

            current_batch_size = (
                validation_images.size(0)
            )

            running_loss += (
                loss.item()
                * current_batch_size
            )

            number_of_samples += (
                current_batch_size
            )

            if batch_index % 50 == 0:

                print(
                    f"validation batch "
                    f"{batch_index}/{len(validation_loader)} "
                    f"loss {loss.item():.6f}"
                )

    validation_loss = (
        running_loss
        / number_of_samples
    )

    tp = torch.cat(
        tp_batches,
        dim=0,
    )

    fp = torch.cat(
        fp_batches,
        dim=0,
    )

    fn = torch.cat(
        fn_batches,
        dim=0,
    )

    tn = torch.cat(
        tn_batches,
        dim=0,
    )

    # ========================================================
    # PER-IMAGE METRICS
    # ========================================================

    (
        per_image_iou_classes,
        per_image_dice_classes,
        per_image_precision_classes,
        per_image_recall_classes,
    ) = compute_metrics(
        tp,
        fp,
        fn,
        tn,
    )

    blood_iou_per_image = (
        per_image_iou_classes[
            :,
            BLOOD_CLASS_INDEX,
        ]
    )

    blood_dice_per_image = (
        per_image_dice_classes[
            :,
            BLOOD_CLASS_INDEX,
        ]
    )

    blood_precision_per_image = (
        per_image_precision_classes[
            :,
            BLOOD_CLASS_INDEX,
        ]
    )

    blood_recall_per_image = (
        per_image_recall_classes[
            :,
            BLOOD_CLASS_INDEX,
        ]
    )

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

    mean_image_precision = (
        blood_precision_per_image
        .mean()
        .item()
    )

    std_image_precision = (
        blood_precision_per_image
        .std(correction=0)
        .item()
    )

    mean_image_recall = (
        blood_recall_per_image
        .mean()
        .item()
    )

    std_image_recall = (
        blood_recall_per_image
        .std(correction=0)
        .item()
    )

    # ========================================================
    # DATASET-LEVEL METRICS
    # ========================================================

    global_tp = tp.sum(
        dim=0
    )

    global_fp = fp.sum(
        dim=0
    )

    global_fn = fn.sum(
        dim=0
    )

    global_tn = tn.sum(
        dim=0
    )

    (
        global_iou_classes,
        global_dice_classes,
        global_precision_classes,
        global_recall_classes,
    ) = compute_metrics(
        global_tp,
        global_fp,
        global_fn,
        global_tn,
    )

    global_iou = global_iou_classes[
        BLOOD_CLASS_INDEX
    ].item()

    global_dice = global_dice_classes[
        BLOOD_CLASS_INDEX
    ].item()

    global_precision = global_precision_classes[
        BLOOD_CLASS_INDEX
    ].item()

    global_recall = global_recall_classes[
        BLOOD_CLASS_INDEX
    ].item()

    selection_score = (
        GLOBAL_DICE_WEIGHT
        * global_dice
        +
        MEAN_IMAGE_DICE_WEIGHT
        * mean_image_dice
    )

    return {
        "loss": validation_loss,
        "global_iou": global_iou,
        "global_dice": global_dice,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "mean_image_iou": mean_image_iou,
        "std_image_iou": std_image_iou,
        "mean_image_dice": mean_image_dice,
        "std_image_dice": std_image_dice,
        "mean_image_precision": mean_image_precision,
        "std_image_precision": std_image_precision,
        "mean_image_recall": mean_image_recall,
        "std_image_recall": std_image_recall,
        "selection_score": selection_score,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run DeepLabV3+ training on the Rabbani dataset.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "This training script is configured for GPU execution."
        )

    if len(sys.argv) >= 2:

        maximum_epochs = int(
            sys.argv[1]
        )

    else:

        maximum_epochs = getattr(
            config_split,
            "DEFAULT_EPOCHS",
            50,
        )

    if maximum_epochs <= 0:
        raise ValueError(
            "The number of epochs must be greater than zero."
        )

    configure_reproducibility(
        RANDOM_SEED
    )

    os.makedirs(
        config_split.MODEL_PRETRAINED_DIR,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        config_split.MODEL_PRETRAINED_DIR,
        (
            f"deeplabv3plus_{SEGMENTATION_MODE}_"
            "best_dice_bleed_seg_"
            f"{ENCODER_NAME}_rab.pth"
        ),
    )

    # ========================================================
    # DATASET AND TRANSFORMS
    # ========================================================

    train_transform = (
        create_bleed_train_transform()
    )

    validation_transform = (
        create_bleed_eval_transform()
    )

    train_dataset = CustomImageDataset(
        config_split.CSV_TRAIN_PATH_V1P0,
        train_transform,
    )

    validation_dataset = CustomImageDataset(
        config_split.CSV_VALID_PATH_V1P0,
        validation_transform,
    )

    if len(train_dataset) == 0:
        raise ValueError(
            "The Rabbani training dataset is empty."
        )

    if len(validation_dataset) == 0:
        raise ValueError(
            "The Rabbani validation dataset is empty."
        )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_generator = torch.Generator()

    train_generator.manual_seed(
        RANDOM_SEED
    )

    validation_generator = torch.Generator()

    validation_generator.manual_seed(
        RANDOM_SEED + 1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,

        # The Rabbani training split contains 525 samples. With batch size 4,
        # drop_last avoids a final batch containing only one image.
        drop_last=True,

        persistent_workers=(
            NUM_WORKERS > 0
        ),
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        worker_init_fn=seed_worker,
        generator=validation_generator,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model().to(
        DEVICE
    )

    # ========================================================
    # LOSS FUNCTIONS
    # ========================================================

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

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model.encoder.parameters()
                ),
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "params": (
                    model.decoder.parameters()
                ),
                "lr": DECODER_LEARNING_RATE,
            },
            {
                "params": (
                    model.segmentation_head.parameters()
                ),
                "lr": DECODER_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

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

    # ========================================================
    # CONFIGURATION SUMMARY
    # ========================================================

    print(
        "======== RABBANI DEEPLABV3+ "
        "TRAINING CONFIGURATION ========"
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
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Encoder: {ENCODER_NAME}"
    )

    print(
        f"Encoder output stride: "
        f"{ENCODER_OUTPUT_STRIDE}"
    )

    print(
        f"Decoder channels: "
        f"{DECODER_CHANNELS}"
    )

    print(
        f"Decoder atrous rates: "
        f"{DECODER_ATROUS_RATES}"
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
            "Training loss: "
            "BCEWithLogitsLoss + DiceLoss"
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
        f"{maximum_epochs}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
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

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    # ========================================================
    # TRAINING STATE
    # ========================================================

    best_selection_score = float(
        "-inf"
    )

    best_global_dice = float(
        "-inf"
    )

    best_mean_image_dice = float(
        "-inf"
    )

    best_epoch = 0
    epochs_without_improvement = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch_index in range(
        maximum_epochs
    ):

        current_epoch = (
            epoch_index + 1
        )

        print(
            f"\n------------ EPOCH: "
            f"{current_epoch}/{maximum_epochs} "
            f"------------"
        )

        training_results = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
            cross_entropy_loss=cross_entropy_loss,
        )

        validation_results = validate_model(
            model=model,
            validation_loader=validation_loader,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
            cross_entropy_loss=cross_entropy_loss,
        )

        current_selection_score = (
            validation_results[
                "selection_score"
            ]
        )

        if not np.isfinite(
            current_selection_score
        ):
            raise FloatingPointError(
                "Non-finite validation selection score detected."
            )

        # ====================================================
        # CHECKPOINT SELECTION
        # ====================================================

        meaningful_improvement = (
            current_selection_score
            >
            best_selection_score
            + MIN_CHECKPOINT_IMPROVEMENT
        )

        if meaningful_improvement:

            best_selection_score = (
                current_selection_score
            )

            best_global_dice = (
                validation_results[
                    "global_dice"
                ]
            )

            best_mean_image_dice = (
                validation_results[
                    "mean_image_dice"
                ]
            )

            best_epoch = current_epoch
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

        else:

            epochs_without_improvement += 1

            print(
                "\nNo meaningful validation improvement."
            )

            print(
                f"Epochs without improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(
            current_selection_score
        )

        encoder_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        decoder_learning_rate = (
            optimizer.param_groups[1]["lr"]
        )

        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        print(
            "\n----- Training -----"
        )

        print(
            f"Loss: "
            f"{training_results['loss']:.6f}"
        )

        print(
            f"Average gradient norm: "
            f"{training_results['gradient_norm']:.6f}"
        )

        print(
            "\n----- Validation -----"
        )

        print(
            f"Loss: "
            f"{validation_results['loss']:.6f}"
        )

        print(
            f"Global IoU: "
            f"{validation_results['global_iou']:.6f}"
        )

        print(
            f"Global Dice: "
            f"{validation_results['global_dice']:.6f}"
        )

        print(
            f"Global precision: "
            f"{validation_results['global_precision']:.6f}"
        )

        print(
            f"Global recall: "
            f"{validation_results['global_recall']:.6f}"
        )

        print(
            f"Mean-image IoU: "
            f"{validation_results['mean_image_iou']:.6f} "
            f"+/- "
            f"{validation_results['std_image_iou']:.6f}"
        )

        print(
            f"Mean-image Dice: "
            f"{validation_results['mean_image_dice']:.6f} "
            f"+/- "
            f"{validation_results['std_image_dice']:.6f}"
        )

        print(
            f"Mean-image precision: "
            f"{validation_results['mean_image_precision']:.6f} "
            f"+/- "
            f"{validation_results['std_image_precision']:.6f}"
        )

        print(
            f"Mean-image recall: "
            f"{validation_results['mean_image_recall']:.6f} "
            f"+/- "
            f"{validation_results['std_image_recall']:.6f}"
        )

        print(
            f"Selection score: "
            f"{current_selection_score:.6f}"
        )

        print(
            f"Encoder learning rate: "
            f"{encoder_learning_rate:.8f}"
        )

        print(
            f"Decoder learning rate: "
            f"{decoder_learning_rate:.8f}"
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Best selection score: "
            f"{best_selection_score:.6f}"
        )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping activated."
            )

            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n======== RABBANI DEEPLABV3+ "
        "TRAINING COMPLETED ========"
    )

    print(
        f"Model: "
        f"{MODEL_NAME}"
    )

    print(
        f"Encoder: "
        f"{ENCODER_NAME}"
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
        f"Best checkpoint: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()