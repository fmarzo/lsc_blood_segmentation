"""
file: evaluate_hemoset_on_rabbani.py

brief:
    Evaluate a segmentation model trained on the Rabbani Bleeding
    Segmentation dataset using the HemoSet test split.

    The model architecture is selected in src/config_split.py through:

        RABBANI_EVALUATION_MODEL = "unet_plus_plus"

    or:

        RABBANI_EVALUATION_MODEL = "deeplabv3plus"

    The segmentation mode is selected through:

        SEGMENTATION_MODE = "binary"

    or:

        SEGMENTATION_MODE = "multiclass"

    Binary evaluation:

    - the model produces one output channel representing blood;
    - the corresponding model was trained using
      BCEWithLogitsLoss + DiceLoss;
    - predictions are obtained using sigmoid followed by
      config_split.BINARY_THRESHOLD;
    - the blood channel index is 0.

    Multiclass evaluation:

    - the model produces two output channels;
    - class 0 represents background;
    - class 1 represents blood;
    - the corresponding model was trained using CrossEntropyLoss;
    - predictions are obtained using argmax;
    - the blood class index is 1.

    This is a zero-shot cross-dataset evaluation:

        training dataset: Rabbani Bleeding Segmentation
        evaluation dataset: HemoSet test split

    Dataset-level metrics are computed after summing TP, FP, FN and TN
    across all HemoSet test images.

    Per-image metrics are computed independently for every image and
    reported using mean and standard deviation.

    Per-video metrics are computed for every video present in the HemoSet
    test split.

usage:
    python -m scripts.rabbani.evaluate_rabbani_on_hemoset
"""

import os
import re

import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import create_eval_transform
from src.hemoset_dataset_v2 import CustomImageDataset


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = (
    config_split.RABBANI_EVALUATION_MODEL
    .strip()
    .lower()
)

SUPPORTED_MODELS = {
    "unet_plus_plus",
    "deeplabv3plus",
}

if MODEL_NAME not in SUPPORTED_MODELS:
    raise ValueError(
        "Unsupported RABBANI_EVALUATION_MODEL value: "
        f"{MODEL_NAME}. Supported values are "
        "'unet_plus_plus' and 'deeplabv3plus'."
    )


ENCODER_NAME = "resnet18"


# ============================================================
# SEGMENTATION CONFIGURATION
# ============================================================

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
        f"{SEGMENTATION_MODE}. Supported values are "
        "'binary' and 'multiclass'."
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


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

BATCH_SIZE = 4
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DEEPLABV3+ CONFIGURATION
# ============================================================

ENCODER_OUTPUT_STRIDE = 16
DECODER_CHANNELS = 256
DECODER_ATROUS_RATES = (12, 24, 36)
UPSAMPLING_FACTOR = 4


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured for GPU evaluation."
    )


# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings.
torch.backends.nnpack.set_flags(False)


# ============================================================
# MODEL HELPERS
# ============================================================

def create_model_and_checkpoint():
    """
    Create the Rabbani-trained architecture selected in config_split.py.

    The number of output channels and checkpoint filename are selected
    according to SEGMENTATION_MODE.
    """
    if MODEL_NAME == "unet_plus_plus":

        model = smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_OUTPUT_CHANNELS,
            activation=None,
        )

        checkpoint_filename = (
            f"unet_plus_plus_{SEGMENTATION_MODE}_"
            "best_dice_bleed_seg_"
            f"{ENCODER_NAME}_rab.pth"
        )

        model_display_name = "U-Net++"

    elif MODEL_NAME == "deeplabv3plus":

        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            encoder_output_stride=ENCODER_OUTPUT_STRIDE,
            decoder_channels=DECODER_CHANNELS,
            decoder_atrous_rates=DECODER_ATROUS_RATES,
            in_channels=3,
            classes=NUM_OUTPUT_CHANNELS,
            activation=None,
            upsampling=UPSAMPLING_FACTOR,
        )

        # The _rab suffix identifies a checkpoint trained on Rabbani.
        checkpoint_filename = (
            f"deeplabv3plus_{SEGMENTATION_MODE}_"
            "best_dice_bleed_seg_"
            f"{ENCODER_NAME}_rab.pth"
        )

        model_display_name = "DeepLabV3+"

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
    Load either a plain model state dictionary or a structured checkpoint.
    """
    if not os.path.isfile(
        checkpoint_path
    ):
        raise FileNotFoundError(
            "Rabbani checkpoint not found: "
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
    Prepare masks according to SEGMENTATION_MODE.

    Binary:
        preserve [B, 1, H, W] and convert values to float.

    Multiclass:
        convert [B, 1, H, W] to [B, H, W] and use integer indices.
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
            "Unexpected model output shape. "
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


def extract_video_number(video_id):
    """
    Extract the numeric component from identifiers such as pig3 or pig11.
    """
    match = re.search(
        r"(\d+)",
        str(video_id),
    )

    if match is None:
        return float("inf")

    return int(
        match.group(1)
    )


# ============================================================
# MODEL AND RABBANI CHECKPOINT
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
# HEMOSET TEST DATASET
# ============================================================

# HemoSet images use the HemoSet evaluation preprocessing.
eval_transform = create_eval_transform()


test_dataset = CustomImageDataset(
    config_split.CSV_TEST_PATH,
    eval_transform,
)


if len(test_dataset) == 0:
    raise ValueError(
        "The HemoSet test dataset is empty: "
        f"{config_split.CSV_TEST_PATH}"
    )


if (
    config_split.VIDEOID_STRING
    not in test_dataset.csv_dirs.columns
):
    raise KeyError(
        "Video ID column not found in the HemoSet test CSV: "
        f"{config_split.VIDEOID_STRING}"
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
    "======== RABBANI MODEL TO HEMOSET "
    "ZERO-SHOT EVALUATION ========"
)

print(
    f"Model: {model_display_name}"
)

print(
    f"Model configuration value: "
    f"{MODEL_NAME}"
)

print(
    f"Encoder: {ENCODER_NAME}"
)


if MODEL_NAME == "deeplabv3plus":

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
    "Training dataset: "
    "Rabbani Bleeding Segmentation"
)

print(
    "Evaluation dataset: "
    "HemoSet test split"
)

print(
    f"Checkpoint: "
    f"{checkpoint_path}"
)

print(
    f"Test CSV: "
    f"{config_split.CSV_TEST_PATH}"
)

print(
    f"Test images: "
    f"{len(test_dataset)}"
)

print(
    f"Test batches: "
    f"{len(test_loader)}"
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

        if batch_index == 0:

            validate_output_shape(
                logits,
                test_masks,
            )

            print(
                f"Input batch shape: "
                f"{tuple(test_images.shape)}"
            )

            print(
                f"Output batch shape: "
                f"{tuple(logits.shape)}"
            )

            print(
                f"Mask batch shape: "
                f"{tuple(test_masks.shape)}"
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
                f"{batch_index}/"
                f"{len(test_loader)}"
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


mean_iou = (
    iou_per_image
    .mean()
    .item()
)

std_iou = (
    iou_per_image
    .std(correction=0)
    .item()
)


mean_dice = (
    dice_per_image
    .mean()
    .item()
)

std_dice = (
    dice_per_image
    .std(correction=0)
    .item()
)


mean_precision = (
    precision_per_image
    .mean()
    .item()
)

std_precision = (
    precision_per_image
    .std(correction=0)
    .item()
)


mean_recall = (
    recall_per_image
    .mean()
    .item()
)

std_recall = (
    recall_per_image
    .std(correction=0)
    .item()
)


# ============================================================
# DATASET-LEVEL METRICS
# ============================================================

global_tp = tp.sum(
    dim=0,
)

global_fp = fp.sum(
    dim=0,
)

global_fn = fn.sum(
    dim=0,
)

global_tn = tn.sum(
    dim=0,
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


total_images = (
    iou_per_image.numel()
)


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


print(
    "\n----- Test image composition -----"
)

print(
    f"Total images: "
    f"{total_images}"
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


# ============================================================
# HEMOSET VIDEO IDENTIFIERS
# ============================================================

test_video_ids = test_dataset.csv_dirs[
    config_split.VIDEOID_STRING
].astype(str)


video_ids = sorted(
    test_video_ids.unique(),
    key=extract_video_number,
)


print(
    "\n----- HemoSet test images per video -----"
)

print(
    test_video_ids.value_counts()
)


# ============================================================
# PER-VIDEO HEMOSET METRICS
# ============================================================

print(
    "\n----- Per-video blood metrics -----"
)


for video_id in video_ids:

    video_mask = torch.tensor(
        (
            test_video_ids
            == str(video_id)
        ).to_numpy(),
        dtype=torch.bool,
    )

    video_image_count = (
        video_mask.sum().item()
    )

    if video_image_count == 0:
        continue


    video_iou_per_image = iou_per_image[
        video_mask
    ]

    video_dice_per_image = dice_per_image[
        video_mask
    ]

    video_precision_per_image = precision_per_image[
        video_mask
    ]

    video_recall_per_image = recall_per_image[
        video_mask
    ]


    video_tp = tp[
        video_mask
    ].sum(
        dim=0
    )

    video_fp = fp[
        video_mask
    ].sum(
        dim=0
    )

    video_fn = fn[
        video_mask
    ].sum(
        dim=0
    )

    video_tn = tn[
        video_mask
    ].sum(
        dim=0
    )


    (
        video_iou_classes,
        video_dice_classes,
        video_precision_classes,
        video_recall_classes,
    ) = compute_metrics(
        video_tp,
        video_fp,
        video_fn,
        video_tn,
    )


    video_global_iou = video_iou_classes[
        BLOOD_CLASS_INDEX
    ].item()

    video_global_dice = video_dice_classes[
        BLOOD_CLASS_INDEX
    ].item()

    video_global_precision = (
        video_precision_classes[
            BLOOD_CLASS_INDEX
        ].item()
    )

    video_global_recall = video_recall_classes[
        BLOOD_CLASS_INDEX
    ].item()


    video_mean_iou = (
        video_iou_per_image
        .mean()
        .item()
    )

    video_std_iou = (
        video_iou_per_image
        .std(correction=0)
        .item()
    )


    video_mean_dice = (
        video_dice_per_image
        .mean()
        .item()
    )

    video_std_dice = (
        video_dice_per_image
        .std(correction=0)
        .item()
    )


    video_mean_precision = (
        video_precision_per_image
        .mean()
        .item()
    )

    video_std_precision = (
        video_precision_per_image
        .std(correction=0)
        .item()
    )


    video_mean_recall = (
        video_recall_per_image
        .mean()
        .item()
    )

    video_std_recall = (
        video_recall_per_image
        .std(correction=0)
        .item()
    )


    print(
        f"\nVideo: "
        f"{video_id}"
    )

    print(
        f"Images: "
        f"{video_image_count}"
    )

    print(
        "Per-image IoU:       "
        f"{video_mean_iou:.4f} "
        f"+/- {video_std_iou:.4f}"
    )

    print(
        "Per-image Dice:      "
        f"{video_mean_dice:.4f} "
        f"+/- {video_std_dice:.4f}"
    )

    print(
        "Per-image Precision: "
        f"{video_mean_precision:.4f} "
        f"+/- {video_std_precision:.4f}"
    )

    print(
        "Per-image Recall:    "
        f"{video_mean_recall:.4f} "
        f"+/- {video_std_recall:.4f}"
    )

    print(
        f"Global IoU:       "
        f"{video_global_iou:.4f}"
    )

    print(
        f"Global Dice:      "
        f"{video_global_dice:.4f}"
    )

    print(
        f"Global Precision: "
        f"{video_global_precision:.4f}"
    )

    print(
        f"Global Recall:    "
        f"{video_global_recall:.4f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n======== RABBANI MODEL TO HEMOSET "
    "ZERO-SHOT RESULTS ========"
)

print(
    f"Model: "
    f"{model_display_name}"
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
    "Training dataset: "
    "Rabbani Bleeding Segmentation"
)

print(
    "Test dataset: "
    "HemoSet"
)

print(
    f"Checkpoint: "
    f"{checkpoint_path}"
)

print(
    f"Test images: "
    f"{total_images}"
)


print(
    "\n----- Dataset-level blood metrics -----"
)

print(
    f"IoU:       "
    f"{global_iou:.4f}"
)

print(
    f"Dice:      "
    f"{global_dice:.4f}"
)

print(
    f"Precision: "
    f"{global_precision:.4f}"
)

print(
    f"Recall:    "
    f"{global_recall:.4f}"
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