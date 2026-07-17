"""
file: final_train.py

brief:
    Final training script for U-Net and U-Net++.

    The script trains a new model from scratch on the union of the original
    training and validation videos.

    The model architecture, encoder, segmentation mode, training videos,
    hyperparameters and number of epochs are selected in config_split.py.

    No validation set is used during this phase. The test videos are never
    loaded. The final checkpoint is saved after the last configured epoch.

usage:
    python -m scripts.final_train
"""

import os
import random

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import create_train_transform
from src.hemoset_dataset_v2 import CustomImageDataset


# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings on the CPU node.
torch.backends.nnpack.set_flags(False)


def set_random_seed(seed):
    """
    Set the random seed for reproducible training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_segmentation_configuration():
    """
    Check that SEGMENTATION_MODE and NUM_CLASSES are consistent.
    """
    mode = config_split.SEGMENTATION_MODE

    if mode not in ["binary", "multiclass"]:
        raise ValueError(
            "SEGMENTATION_MODE must be 'binary' or 'multiclass'."
        )

    if mode == "binary" and config_split.NUM_CLASSES != 1:
        raise ValueError(
            "Binary segmentation requires NUM_CLASSES = 1."
        )

    if mode == "multiclass" and config_split.NUM_CLASSES < 2:
        raise ValueError(
            "Multiclass segmentation requires NUM_CLASSES >= 2."
        )


def validate_final_video_split():
    """
    Check that final training and test videos do not overlap.
    """
    final_train_video_ids = config_split.FINAL_TRAIN_VIDEO_ID
    test_video_ids = config_split.TEST_VIDEO_ID

    if len(final_train_video_ids) == 0:
        raise ValueError(
            "FINAL_TRAIN_VIDEO_ID cannot be empty."
        )

    if len(final_train_video_ids) != len(
        set(final_train_video_ids)
    ):
        raise ValueError(
            "Duplicate video IDs found in FINAL_TRAIN_VIDEO_ID."
        )

    overlap = (
        set(final_train_video_ids)
        & set(test_video_ids)
    )

    if overlap:
        raise ValueError(
            "Final training and test videos overlap: "
            f"{sorted(overlap)}"
        )


def get_final_num_epochs(model_name, encoder_name):
    """
    Retrieve the selected number of epochs for the model-encoder pair.
    """
    try:
        n_epochs = config_split.FINAL_NUM_EPOCHS[
            model_name
        ][encoder_name]

    except KeyError as error:
        raise KeyError(
            "Missing final epoch configuration for "
            f"model='{model_name}' and encoder='{encoder_name}'."
        ) from error

    if not isinstance(n_epochs, int) or n_epochs <= 0:
        raise ValueError(
            "The selected final number of epochs must be "
            f"a positive integer. Received: {n_epochs}"
        )

    return n_epochs


def create_final_train_csv():
    """
    Create a CSV containing only the videos selected for final training.
    """
    complete_df = pd.read_csv(
        config_split.CSV_FILE_PATH
    )

    video_id_column = (
        config_split.CSV_VIDEO_ID_COLUMN
    )

    if video_id_column not in complete_df.columns:
        raise KeyError(
            f"Column '{video_id_column}' not found in "
            f"{config_split.CSV_FILE_PATH}. "
            f"Available columns: {list(complete_df.columns)}"
        )

    available_video_ids = set(
        complete_df[
            video_id_column
        ].astype(str).unique()
    )

    requested_video_ids = {
        str(video_id)
        for video_id
        in config_split.FINAL_TRAIN_VIDEO_ID
    }

    missing_video_ids = (
        requested_video_ids
        - available_video_ids
    )

    if missing_video_ids:
        raise ValueError(
            "The following final training videos were not "
            f"found in the complete CSV: {sorted(missing_video_ids)}"
        )

    final_train_df = complete_df[
        complete_df[
            video_id_column
        ].astype(str).isin(requested_video_ids)
    ].copy()

    if final_train_df.empty:
        raise ValueError(
            "The final training dataframe is empty."
        )

    os.makedirs(
        os.path.dirname(
            config_split.CSV_FINAL_TRAIN_PATH
        ),
        exist_ok=True,
    )

    final_train_df.to_csv(
        config_split.CSV_FINAL_TRAIN_PATH,
        index=False,
    )

    return final_train_df


def create_model(model_name, encoder_name):
    """
    Create U-Net or U-Net++ with the configured encoder.
    """
    model_arguments = {
        "encoder_name": encoder_name,
        "encoder_weights": (
            config_split.FINAL_ENCODER_WEIGHTS
        ),
        "in_channels": 3,
        "classes": config_split.NUM_CLASSES,
    }

    if model_name == "unet":
        model = smp.Unet(
            **model_arguments
        )

    elif model_name == "unet_plus_plus":
        model = smp.UnetPlusPlus(
            **model_arguments
        )

    else:
        raise ValueError(
            "FINAL_MODEL_NAME must be "
            "'unet' or 'unet_plus_plus'."
        )

    return model.to("cuda")


def prepare_mask(mask, mode):
    """
    Prepare the mask for binary or multiclass segmentation.
    """
    if mode == "binary":
        return mask.float().to(
            "cuda",
            non_blocking=True,
        )

    return torch.squeeze(
        mask,
        dim=1,
    ).long().to(
        "cuda",
        non_blocking=True,
    )


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured for GPU training."
    )


validate_segmentation_configuration()
validate_final_video_split()

set_random_seed(
    config_split.FINAL_RANDOM_SEED
)


model_name = (
    config_split.FINAL_MODEL_NAME
    .strip()
    .lower()
)

encoder_name = (
    config_split.FINAL_ENCODER_NAME
    .strip()
    .lower()
)

n_epochs = get_final_num_epochs(
    model_name,
    encoder_name,
)


# Create the CSV containing the seven final training videos.
final_train_df = create_final_train_csv()


# Use training augmentations on the complete final training set.
train_transform = create_train_transform()


final_train_ds = CustomImageDataset(
    config_split.CSV_FINAL_TRAIN_PATH,
    train_transform,
)


data_loader_generator = torch.Generator()

data_loader_generator.manual_seed(
    config_split.FINAL_RANDOM_SEED
)


final_train_dl = DataLoader(
    final_train_ds,
    batch_size=config_split.FINAL_BATCH_SIZE,
    num_workers=config_split.FINAL_NUM_WORKERS,
    shuffle=True,
    pin_memory=True,
    generator=data_loader_generator,
)


if len(final_train_dl) == 0:
    raise ValueError(
        "The final training DataLoader is empty."
    )


# Create a new model from the configured ImageNet encoder weights.
model = create_model(
    model_name,
    encoder_name,
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config_split.FINAL_LEARNING_RATE,
)


scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=config_split.FINAL_LR_MILESTONES,
    gamma=config_split.FINAL_LR_GAMMA,
)


# Initialize only the losses required by the selected mode.
if config_split.SEGMENTATION_MODE == "binary":

    bce_loss = (
        torch.nn.BCEWithLogitsLoss()
        .to("cuda")
    )

    dice_loss = smp.losses.DiceLoss(
        mode="binary",
        from_logits=True,
    ).to("cuda")

    ce_loss = None

else:

    bce_loss = None
    dice_loss = None

    ce_loss = (
        torch.nn.CrossEntropyLoss()
        .to("cuda")
    )


os.makedirs(
    config_split.MODEL_PRETRAINED_DIR,
    exist_ok=True,
)


checkpoint_filename = (
    f"{model_name}_"
    f"{config_split.SEGMENTATION_MODE}_"
    f"final_{encoder_name}_"
    f"epoch_{n_epochs}.pth"
)


checkpoint_path = os.path.join(
    config_split.MODEL_PRETRAINED_DIR,
    checkpoint_filename,
)


print("======== FINAL TRAINING CONFIGURATION ========")

print(
    f"Model: {model_name}"
)

print(
    f"Encoder: {encoder_name}"
)

print(
    "Encoder weights: "
    f"{config_split.FINAL_ENCODER_WEIGHTS}"
)

print(
    "Segmentation mode: "
    f"{config_split.SEGMENTATION_MODE}"
)

print(
    f"Output classes: {config_split.NUM_CLASSES}"
)

print(
    f"Epochs: {n_epochs}"
)

print(
    "Learning rate: "
    f"{config_split.FINAL_LEARNING_RATE}"
)

print(
    f"Batch size: {config_split.FINAL_BATCH_SIZE}"
)

print(
    f"Training samples: {len(final_train_df)}"
)

print(
    "Training videos: "
    f"{config_split.FINAL_TRAIN_VIDEO_ID}"
)

print(
    "Final training CSV: "
    f"{config_split.CSV_FINAL_TRAIN_PATH}"
)

print(
    f"Checkpoint: {checkpoint_path}"
)


for epoch in range(n_epochs):

    model.train()

    train_loss_sum = 0.0
    train_sample_count = 0

    print(
        f"------------ EPOCH: "
        f"{epoch + 1}/{n_epochs} ------------"
    )

    for batch_index, (
        train_img,
        train_mask,
    ) in enumerate(final_train_dl):

        optimizer.zero_grad(
            set_to_none=True
        )

        train_img = train_img.to(
            "cuda",
            non_blocking=True,
        )

        train_mask = prepare_mask(
            train_mask,
            config_split.SEGMENTATION_MODE,
        )

        logits = model(
            train_img
        )

        if (
            config_split.SEGMENTATION_MODE
            == "binary"
        ):

            loss_train_value = (
                bce_loss(
                    logits,
                    train_mask,
                )
                +
                dice_loss(
                    logits,
                    train_mask,
                )
            )

        else:

            loss_train_value = ce_loss(
                logits,
                train_mask,
            )

        loss_train_value.backward()

        optimizer.step()

        batch_size = train_img.size(0)

        train_loss_sum += (
            loss_train_value.item()
            * batch_size
        )

        train_sample_count += batch_size

        if batch_index % 50 == 0:
            print(
                f"batch "
                f"{batch_index}/{len(final_train_dl)} "
                f"loss_train "
                f"{loss_train_value.item():.6f}"
            )

    average_train_loss = (
        train_loss_sum
        / train_sample_count
    )

    scheduler.step()

    current_learning_rate = (
        scheduler.get_last_lr()[0]
    )

    print(
        f"avg_train_loss "
        f"{average_train_loss:.6f}"
    )

    print(
        f"learning_rate "
        f"{current_learning_rate}"
    )


# No validation-based model selection is performed here.
# Save only the model obtained after the configured final epoch.
torch.save(
    model.state_dict(),
    checkpoint_path,
)


print(
    "\n======== FINAL TRAINING COMPLETED ========"
)

print(
    f"Model: {model_name}"
)

print(
    f"Encoder: {encoder_name}"
)

print(
    "Segmentation mode: "
    f"{config_split.SEGMENTATION_MODE}"
)

print(
    f"Epochs: {n_epochs}"
)

print(
    f"Training samples: {len(final_train_df)}"
)

print(
    f"Checkpoint saved to: {checkpoint_path}"
)

