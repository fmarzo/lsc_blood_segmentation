"""
file: evaluate_hemoset_on_rabbani.py

brief:
    Evaluate a binary segmentation model trained on HemoSet using the
    Rabbani Bleeding Segmentation v1.0 test split.

    The architecture is selected through:

        config_split.MODEL_TO_EVALUATE

    Supported values:

        "unet"
        "unet_plus_plus"

    The remaining model configuration is fixed:

    - encoder: ResNet-18
    - segmentation mode: binary
    - output channels: 1
    - blood threshold: 0.50

    The corresponding checkpoints are:

        unet_binary_best_resnet18.pth

        unet_plus_plus_binary_best_resnet18.pth

    This is a zero-shot cross-dataset evaluation:

        training dataset: HemoSet
        test dataset: Rabbani

usage:
    python -m scripts.rabbani.evaluate_hemoset_on_rabbani
"""

import os

import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import create_bleed_eval_transform
from src.hemoset_dataset_v2 import CustomImageDataset


# ============================================================
# FIXED MODEL CONFIGURATION
# ============================================================

MODEL_NAME = (
    config_split.MODEL_TO_EVALUATE
    .strip()
    .lower()
)

ENCODER_NAME = "resnet18"

SEGMENTATION_MODE = "binary"
NUM_OUTPUT_CHANNELS = 1

BINARY_THRESHOLD = 0.50
BLOOD_CLASS_INDEX = 0

BATCH_SIZE = 4
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings.
torch.backends.nnpack.set_flags(False)


# ============================================================
# MODEL HELPERS
# ============================================================

def create_model_and_checkpoint():
    """
    Create the binary ResNet-18 model selected through MODEL_TO_EVALUATE and
    return its checkpoint path and display name.
    """
    if MODEL_NAME == "unet":

        model = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_OUTPUT_CHANNELS,
            activation=None,
        )

        checkpoint_filename = (
            "unet_binary_best_resnet18.pth"
        )

        model_display_name = "U-Net"

    elif MODEL_NAME == "unet_plus_plus":

        model = smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_OUTPUT_CHANNELS,
            activation=None,
        )

        checkpoint_filename = (
            "unet_plus_plus_binary_best_resnet18.pth"
        )

        model_display_name = "U-Net++"

    else:
        raise ValueError(
            "Unsupported MODEL_TO_EVALUATE value: "
            f"{MODEL_NAME}. "
            "Supported values are 'unet' and 'unet_plus_plus'."
        )

    checkpoint_path = os.path.join(
        config_split.MODEL_PRETRAINED_DIR,
        checkpoint_filename,
    )

    return (
        model.to(DEVICE),
        checkpoint_path,
        model_display_name,
    )


def load_checkpoint(
    model,
    checkpoint_path,
):
    """
    Load either a plain state dictionary or a structured checkpoint.
    """
    if not os.path.isfile(
        checkpoint_path
    ):
        raise FileNotFoundError(
            "HemoSet checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()


# ============================================================
# SEGMENTATION HELPERS
# ============================================================

def prepare_mask(mask):
    """
    Prepare a binary mask while preserving its [B, 1, H, W] shape.
    """
    return mask.float().to(
        DEVICE,
        non_blocking=True,
    )


def get_predictions(logits):
    """
    Convert binary logits into a binary blood segmentation map.
    """
    probabilities = torch.sigmoid(
        logits
    )

    return (
        probabilities
        >= BINARY_THRESHOLD
    ).long()


def get_segmentation_stats(
    predictions,
    mask,
):
    """
    Compute TP, FP, FN and TN independently for every image.
    """
    return smp.metrics.get_stats(
        predictions,
        mask.long(),
        mode=SEGMENTATION_MODE,
    )


def compute_metrics(
    tp,
    fp,
    fn,
    tn,
):
    """
    Compute IoU, Dice, precision and recall.
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
# CUDA CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured for GPU evaluation."
    )


# ============================================================
# MODEL AND HEMOSET CHECKPOINT
# ============================================================

(
    model,
    checkpoint_path,
    model_display_name,
) = create_model_and_checkpoint()


load_checkpoint(
    model=model,
    checkpoint_path=checkpoint_path,
)


# ============================================================
# RABBANI TEST DATASET
# ============================================================

eval_transform = (
    create_bleed_eval_transform()
)


test_dataset = CustomImageDataset(
    config_split.CSV_TEST_PATH_V1P0,
    eval_transform,
)


if len(test_dataset) == 0:
    raise ValueError(
        "The Rabbani test dataset is empty: "
        f"{config_split.CSV_TEST_PATH_V1P0}"
    )


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(
        NUM_WORKERS > 0
    ),
)


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

print(
    "======== HEMOSET MODEL TO RABBANI "
    "ZERO-SHOT EVALUATION ========"
)

print(
    f"Model: {model_display_name}"
)

print(
    f"MODEL_TO_EVALUATE: {MODEL_NAME}"
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
    f"Blood threshold: "
    f"{BINARY_THRESHOLD:.2f}"
)

print(
    "Training dataset: HemoSet"
)

print(
    "Evaluation dataset: Rabbani test split"
)

print(
    f"Checkpoint: {checkpoint_path}"
)

print(
    f"Test CSV: "
    f"{config_split.CSV_TEST_PATH_V1P0}"
)

print(
    f"Test images: {len(test_dataset)}"
)

print(
    f"Test batches: {len(test_loader)}"
)


# ============================================================
# INFERENCE
# ============================================================

tp_batches = []
fp_batches = []
fn_batches = []
tn_batches = []


with torch.inference_mode():

    for batch_index, (
        test_images,
        test_masks,
    ) in enumerate(test_loader):

        test_images = test_images.to(
            DEVICE,
            non_blocking=True,
        )

        test_masks = prepare_mask(
            test_masks
        )

        logits = model(
            test_images
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
            test_masks,
        )

        # Preserve one row for every Rabbani test image.
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
                f"Evaluated batch "
                f"{batch_index}/{len(test_loader)}"
            )


# Join all batches while preserving one row for each test image.
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


# ============================================================
# PER-IMAGE METRICS
# ============================================================

(
    iou_classes,
    dice_classes,
    precision_classes,
    recall_classes,
) = compute_metrics(
    tp,
    fp,
    fn,
    tn,
)


# Binary mode has a single foreground channel at index zero.
iou_per_image = iou_classes[
    :,
    BLOOD_CLASS_INDEX,
]

dice_per_image = dice_classes[
    :,
    BLOOD_CLASS_INDEX,
]

precision_per_image = precision_classes[
    :,
    BLOOD_CLASS_INDEX,
]

recall_per_image = recall_classes[
    :,
    BLOOD_CLASS_INDEX,
]


mean_iou = iou_per_image.mean().item()

std_iou = iou_per_image.std(
    correction=0
).item()


mean_dice = dice_per_image.mean().item()

std_dice = dice_per_image.std(
    correction=0
).item()


mean_precision = (
    precision_per_image
    .mean()
    .item()
)

std_precision = precision_per_image.std(
    correction=0
).item()


mean_recall = (
    recall_per_image
    .mean()
    .item()
)

std_recall = recall_per_image.std(
    correction=0
).item()


# ============================================================
# DATASET-LEVEL METRICS
# ============================================================

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


# ============================================================
# TEST SET COMPOSITION
# ============================================================

blood_gt_pixels = (
    tp[:, BLOOD_CLASS_INDEX]
    + fn[:, BLOOD_CLASS_INDEX]
)

blood_pred_pixels = (
    tp[:, BLOOD_CLASS_INDEX]
    + fp[:, BLOOD_CLASS_INDEX]
)


images_with_blood_mask = (
    blood_gt_pixels > 0
)

empty_images_mask = (
    blood_gt_pixels == 0
)


total_images = iou_per_image.numel()


images_with_blood = (
    images_with_blood_mask
    .sum()
    .item()
)

empty_images = (
    empty_images_mask
    .sum()
    .item()
)


correct_empty_predictions = (
    empty_images_mask
    & (blood_pred_pixels == 0)
).sum().item()


empty_images_with_false_positives = (
    empty_images_mask
    & (blood_pred_pixels > 0)
).sum().item()


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n----- Test image composition -----"
)

print(
    f"Total images: {total_images}"
)

print(
    f"Images with blood: "
    f"{images_with_blood}"
)

print(
    f"Images without blood: "
    f"{empty_images}"
)


print(
    "\n----- Empty image predictions -----"
)

print(
    "Correctly predicted as empty: "
    f"{correct_empty_predictions}"
)

print(
    "Empty images with false-positive blood: "
    f"{empty_images_with_false_positives}"
)


print(
    "\n======== HEMOSET TO RABBANI "
    "ZERO-SHOT RESULTS ========"
)

print(
    f"Model: {model_display_name}"
)

print(
    f"Encoder: {ENCODER_NAME}"
)

print(
    f"Segmentation mode: "
    f"{SEGMENTATION_MODE}"
)

print(
    "Training dataset: HemoSet"
)

print(
    "Test dataset: Rabbani"
)

print(
    f"Checkpoint: {checkpoint_path}"
)

print(
    f"Test images: {total_images}"
)


print(
    "\n----- Dataset-level blood metrics -----"
)

print(
    f"IoU:       {global_iou:.4f}"
)

print(
    f"Dice:      {global_dice:.4f}"
)

print(
    f"Precision: {global_precision:.4f}"
)

print(
    f"Recall:    {global_recall:.4f}"
)


print(
    "\n----- Per-image blood metrics -----"
)

print(
    f"IoU:       "
    f"{mean_iou:.4f} "
    f"+/- {std_iou:.4f}"
)

print(
    f"Dice:      "
    f"{mean_dice:.4f} "
    f"+/- {std_dice:.4f}"
)

print(
    f"Precision: "
    f"{mean_precision:.4f} "
    f"+/- {std_precision:.4f}"
)

print(
    f"Recall:    "
    f"{mean_recall:.4f} "
    f"+/- {std_recall:.4f}"
)