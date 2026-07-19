"""
file: evaluate_deeplabv3plus_hemoset.py

brief:
    Evaluate the best DeepLabV3+ ResNet-18 checkpoint trained on HemoSet.

    The segmentation mode is selected in src/config_split.py through:

        SEGMENTATION_MODE = "binary"

    or:

        SEGMENTATION_MODE = "multiclass"

    Binary evaluation:

    - the network produces one output channel representing blood;
    - the corresponding model was trained using
      BCEWithLogitsLoss + DiceLoss;
    - predictions are obtained using sigmoid followed by
      config_split.BINARY_THRESHOLD;
    - the blood channel index is 0.

    Multiclass evaluation:

    - the network produces two output channels;
    - class 0 represents background;
    - class 1 represents blood;
    - the corresponding model was trained using CrossEntropyLoss;
    - predictions are obtained using argmax;
    - the blood class index is 1.

    Fixed architecture:

    - DeepLabV3+
    - ResNet-18 encoder
    - encoder output stride 16
    - decoder channels 256
    - atrous rates 12, 24 and 36

    The checkpoint filename is selected automatically according to the
    segmentation mode:

        deeplabv3plus_binary_best_resnet18_hemo.pth

    or:

        deeplabv3plus_multiclass_best_resnet18_hemo.pth

    Evaluation is performed on the HemoSet test split without random data
    augmentation and without gradient computation.

    Dataset-level metrics are computed after summing TP, FP, FN and TN
    across every test image.

    Per-image metrics are computed independently for every image and are
    reported as mean and standard deviation.

    Per-video metrics are reported for every pig contained in the HemoSet
    test CSV.

usage:
    python -m scripts.evaluate_deeplabv3plus_hemoset
"""

import re
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import create_eval_transform
from src.hemoset_dataset_v2 import CustomImageDataset


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "deeplabv3plus"
MODEL_DISPLAY_NAME = "DeepLabV3+"

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

    TRAINING_LOSS_NAME = (
        "BCEWithLogitsLoss + DiceLoss"
    )

else:

    NUM_OUTPUT_CHANNELS = 2

    BACKGROUND_CLASS_INDEX = 0
    BLOOD_CLASS_INDEX = 1

    BINARY_THRESHOLD = None

    TRAINING_LOSS_NAME = (
        "CrossEntropyLoss"
    )


# ============================================================
# DEEPLABV3+ CONFIGURATION
# ============================================================

ENCODER_OUTPUT_STRIDE = 16
DECODER_CHANNELS = 256
DECODER_ATROUS_RATES = (12, 24, 36)
UPSAMPLING_FACTOR = 4


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

BATCH_SIZE = 4
NUM_WORKERS = 2

VIDEO_ID_COLUMN = "video_id"


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PATHS
# ============================================================

# The script is expected inside:
#
#     lab/scripts/evaluate_deeplabv3plus_hemoset.py
#
# Therefore parents[1] identifies the lab directory.
LAB_DIRECTORY = Path(
    __file__
).resolve().parents[1]


HEMOSET_TEST_CSV_PATH = (
    LAB_DIRECTORY
    / "splits"
    / "test.csv"
)


MODEL_PRETRAINED_DIRECTORY = Path(
    "/work/cvcs2026/latent_space_cowboys/model_pretrained"
)


CHECKPOINT_PATH = (
    MODEL_PRETRAINED_DIRECTORY
    / (
        f"deeplabv3plus_{SEGMENTATION_MODE}_"
        f"best_{ENCODER_NAME}_hemo.pth"
    )
)


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This evaluation script is configured for GPU execution."
    )


# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings.
torch.backends.nnpack.set_flags(False)


# ============================================================
# MODEL HELPERS
# ============================================================

def create_model():
    """
    Create the exact DeepLabV3+ architecture used during HemoSet training.

    The number of output channels is selected using SEGMENTATION_MODE.
    ImageNet weights are not loaded because the complete trained state
    dictionary is restored from the checkpoint.
    """
    return smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        encoder_output_stride=ENCODER_OUTPUT_STRIDE,
        decoder_channels=DECODER_CHANNELS,
        decoder_atrous_rates=DECODER_ATROUS_RATES,
        in_channels=3,
        classes=NUM_OUTPUT_CHANNELS,
        activation=None,
        upsampling=UPSAMPLING_FACTOR,
    ).to(
        DEVICE
    )


def load_checkpoint(
    model,
    checkpoint_path,
):
    """
    Load a plain state dictionary or a structured checkpoint.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "DeepLabV3+ HemoSet checkpoint not found: "
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

    Binary mode:
        preserve [B, 1, H, W] and convert values to float.

    Multiclass mode:
        convert [B, 1, H, W] to [B, H, W] and use integer class indices.
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


def validate_output_shape(
    logits,
    masks,
):
    """
    Verify that model outputs and masks have compatible dimensions.
    """
    if SEGMENTATION_MODE == "binary":

        expected_shape = tuple(
            masks.shape
        )

    else:

        expected_shape = (
            masks.shape[0],
            NUM_OUTPUT_CHANNELS,
            masks.shape[1],
            masks.shape[2],
        )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected DeepLabV3+ output shape. "
            f"Received {tuple(logits.shape)}, "
            f"expected {expected_shape} for "
            f"{SEGMENTATION_MODE} segmentation."
        )


def get_segmentation_stats(
    predictions,
    masks,
):
    """
    Compute TP, FP, FN and TN independently for every image.
    """
    if SEGMENTATION_MODE == "binary":

        return smp.metrics.get_stats(
            predictions,
            masks.long(),
            mode="binary",
        )

    return smp.metrics.get_stats(
        predictions,
        masks,
        mode="multiclass",
        num_classes=NUM_OUTPUT_CHANNELS,
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
# CHECK INPUT FILES
# ============================================================

if not HEMOSET_TEST_CSV_PATH.is_file():
    raise FileNotFoundError(
        "HemoSet test CSV not found: "
        f"{HEMOSET_TEST_CSV_PATH}"
    )


if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        "DeepLabV3+ HemoSet checkpoint not found: "
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# MODEL AND CHECKPOINT
# ============================================================

deeplabv3plus = create_model()


load_checkpoint(
    model=deeplabv3plus,
    checkpoint_path=CHECKPOINT_PATH,
)


# ============================================================
# HEMOSET TEST DATASET
# ============================================================

# No random augmentation is applied during evaluation.
eval_transform = create_eval_transform()


test_dataset = CustomImageDataset(
    str(HEMOSET_TEST_CSV_PATH),
    eval_transform,
)


if len(test_dataset) == 0:
    raise ValueError(
        "The HemoSet test dataset is empty: "
        f"{HEMOSET_TEST_CSV_PATH}"
    )


if VIDEO_ID_COLUMN not in test_dataset.csv_dirs.columns:
    raise KeyError(
        f"Column '{VIDEO_ID_COLUMN}' was not found in the "
        "HemoSet test CSV. Available columns: "
        f"{list(test_dataset.csv_dirs.columns)}"
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
    "======== HEMOSET DEEPLABV3+ EVALUATION ========"
)

print(
    f"Model: {MODEL_DISPLAY_NAME}"
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

print(
    f"Training loss: "
    f"{TRAINING_LOSS_NAME}"
)


if SEGMENTATION_MODE == "binary":

    print(
        f"Binary threshold: "
        f"{BINARY_THRESHOLD:.2f}"
    )


print(
    f"Checkpoint: "
    f"{CHECKPOINT_PATH}"
)

print(
    f"Test CSV: "
    f"{HEMOSET_TEST_CSV_PATH}"
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

        logits = deeplabv3plus(
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

        # Preserve one row for every test image.
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
# TEST IMAGES PER VIDEO
# ============================================================

test_video_ids = test_dataset.csv_dirs[
    VIDEO_ID_COLUMN
].astype(str)


video_ids = sorted(
    test_video_ids.unique(),
    key=extract_video_number,
)


print(
    "\n----- Test images per video -----"
)

print(
    test_video_ids.value_counts()
)


# ============================================================
# PER-VIDEO METRICS
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
    "\n======== HEMOSET DEEPLABV3+ TEST RESULTS ========"
)

print(
    f"Model: "
    f"{MODEL_DISPLAY_NAME}"
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
    f"Training loss: "
    f"{TRAINING_LOSS_NAME}"
)


if SEGMENTATION_MODE == "binary":

    print(
        f"Binary threshold: "
        f"{BINARY_THRESHOLD:.2f}"
    )


print(
    f"Checkpoint: "
    f"{CHECKPOINT_PATH}"
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