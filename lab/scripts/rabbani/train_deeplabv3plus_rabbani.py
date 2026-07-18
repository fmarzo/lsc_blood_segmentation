"""
file: train_deeplabv3plus_rabbani.py

brief:
    Train a DeepLabV3+ segmentation network on the Rabbani bleeding
    segmentation dataset.

    Fixed configuration:

    - DeepLabV3+
    - ResNet-18 encoder pretrained on ImageNet
    - multiclass segmentation
    - two output classes
    - CrossEntropyLoss
    - physical batch size 4

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

SEGMENTATION_MODE = "multiclass"
NUM_CLASSES = 2

BACKGROUND_CLASS_INDEX = 0
BLOOD_CLASS_INDEX = 1

ENCODER_OUTPUT_STRIDE = 16
DECODER_CHANNELS = 256
DECODER_ATROUS_RATES = (12, 24, 36)

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

EPSILON = 1e-12


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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def configure_reproducibility(seed):
    """
    Seed Python, NumPy and PyTorch.

    Exact deterministic CUDA execution is not enabled because some
    segmentation operations do not have deterministic implementations.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """
    Seed each DataLoader worker.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ============================================================
# MODEL HELPERS
# ============================================================

def create_model():
    """
    Create the DeepLabV3+ ResNet-18 segmentation model.
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
        upsampling=4,
    )


def freeze_encoder_batch_norm_statistics(model):
    """
    Freeze the running statistics of the pretrained encoder BatchNorm layers.

    The affine BatchNorm parameters remain trainable.
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


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    loss_function,
    optimizer,
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

    optimizer.zero_grad(
        set_to_none=True
    )

    for batch_index, (
        train_images,
        train_masks,
    ) in enumerate(train_loader):

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

        loss = loss_function(
            logits,
            train_masks,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss detected "
                f"at batch {batch_index}: {loss.item()}"
            )

        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_MAX_NORM,
        )

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        current_batch_size = train_images.size(
            0
        )

        running_loss += (
            loss.item()
            * current_batch_size
        )

        number_of_samples += (
            current_batch_size
        )

        gradient_norm_sum += float(
            gradient_norm
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
    loss_function,
):
    """
    Evaluate the model on the validation split.

    Metrics are computed for the blood class only.
    """
    model.eval()

    running_loss = 0.0
    number_of_samples = 0

    global_true_positive = 0.0
    global_false_positive = 0.0
    global_false_negative = 0.0

    per_image_iou_sum = 0.0
    per_image_iou_squared_sum = 0.0

    per_image_dice_sum = 0.0
    per_image_dice_squared_sum = 0.0

    number_of_images = 0

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

            loss = loss_function(
                logits,
                validation_masks,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite validation loss detected "
                    f"at batch {batch_index}: {loss.item()}"
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
                (
                    true_positive
                    / iou_denominator
                ),
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
                (
                    2.0
                    * true_positive
                    / dice_denominator
                ),
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

            running_loss += (
                loss.item()
                * current_batch_size
            )

            number_of_samples += (
                current_batch_size
            )

            number_of_images += (
                current_batch_size
            )

    validation_loss = (
        running_loss
        / number_of_samples
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
        / number_of_images
    )

    mean_image_dice = (
        per_image_dice_sum
        / number_of_images
    )

    image_iou_variance = max(
        0.0,
        (
            per_image_iou_squared_sum
            / number_of_images
        )
        - mean_image_iou ** 2,
    )

    image_dice_variance = max(
        0.0,
        (
            per_image_dice_squared_sum
            / number_of_images
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
        "selection_score": selection_score,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run DeepLabV3+ training.
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
        maximum_epochs = 50

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
            "deeplabv3plus_multiclass_"
            "best_dice_bleed_seg_"
            "resnet18.pth"
        ),
    )

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

        # The training split contains 525 samples. With batch size 4,
        # drop_last avoids a final BatchNorm batch containing one image.
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

    model = create_model().to(
        DEVICE
    )

    loss_function = (
        torch.nn.CrossEntropyLoss()
        .to(DEVICE)
    )

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

    print(
        "======== RABBANI DEEPLABV3+ TRAINING CONFIGURATION ========"
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    print(
        f"Training batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(validation_loader)}"
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
        f"Decoder atrous rates: "
        f"{DECODER_ATROUS_RATES}"
    )

    print(
        f"Segmentation mode: "
        f"{SEGMENTATION_MODE}"
    )

    print(
        f"Number of classes: {NUM_CLASSES}"
    )

    print(
        f"Maximum epochs: {maximum_epochs}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
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

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    best_selection_score = float(
        "-inf"
    )

    best_epoch = 0
    epochs_without_improvement = 0

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
            loss_function=loss_function,
            optimizer=optimizer,
        )

        validation_results = validate_model(
            model=model,
            validation_loader=validation_loader,
            loss_function=loss_function,
        )

        current_selection_score = (
            validation_results[
                "selection_score"
            ]
        )

        scheduler.step(
            current_selection_score
        )

        encoder_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        decoder_learning_rate = (
            optimizer.param_groups[1]["lr"]
        )

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

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping activated."
            )

            break

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


if __name__ == "__main__":
    main()