"""
file: evaluate_rabbani_on_hemoset.py

brief:
    This script evaluates a U-Net model trained on the Rabbani Bleed Seg
    dataset using the HemoSet test split.

    The model configuration is fixed to:

    - architecture: U-Net
    - encoder: ResNet-18
    - segmentation mode: multiclass
    - number of classes: 2
    - class 0: background
    - class 1: blood

    The checkpoint was selected according to the highest validation Dice
    obtained during training on the Rabbani dataset.

    No HemoSet image is used during model training. Therefore, this script
    performs an inverse zero-shot cross-dataset evaluation:

        training dataset: Rabbani Bleed Seg
        test dataset: HemoSet

    Dataset-level metrics are computed by summing TP, FP, FN and TN across all
    test images before calculating the metrics.

    Per-image metrics are computed independently for every test image and
    reported using mean and standard deviation.

    Per-video metrics are also reported for every pig included in the HemoSet
    test split.

usage:
    python -m scripts.rabbani.evaluate_rabbani_on_hemoset
"""

import os

import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import create_eval_transform
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

BATCH_SIZE = 4
NUM_WORKERS = 2


# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings on the CPU node.
torch.backends.nnpack.set_flags(False)


def prepare_mask(mask):
    """
    Convert masks from [B, 1, H, W] to [B, H, W] and prepare them for
    multiclass evaluation.
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
    Compute TP, FP, FN and TN independently for every image and class.
    """
    return smp.metrics.get_stats(
        predictions,
        mask,
        mode=SEGMENTATION_MODE,
        num_classes=NUM_CLASSES,
    )


def compute_metrics(tp, fp, fn, tn):
    """
    Compute IoU, Dice, precision and recall for every class.
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

    return iou, dice, precision, recall


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured for GPU evaluation."
    )


# ============================================================
# MODEL AND RABBANI CHECKPOINT
# ============================================================

model = smp.Unet(
    encoder_name=ENCODER_NAME,
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
).to("cuda")


checkpoint_path = os.path.join(
    config_split.MODEL_PRETRAINED_DIR,
    (
        "unet_multiclass_"
        "best_dice_bleed_seg_"
        "resnet18.pth"
    ),
)


if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
        "Rabbani checkpoint not found: "
        f"{checkpoint_path}"
    )


model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location="cuda",
    )
)


model.eval()


# ============================================================
# HEMOSET TEST DATASET
# ============================================================

# HemoSet images already have the expected spatial resolution.
eval_transform = create_eval_transform()


test_ds = CustomImageDataset(
    config_split.CSV_TEST_PATH,
    eval_transform,
)


if len(test_ds) == 0:
    raise ValueError(
        "The HemoSet test dataset is empty: "
        f"{config_split.CSV_TEST_PATH}"
    )


test_hemo_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
)


print(
    "======== RABBANI TO HEMOSET ZERO-SHOT EVALUATION ========"
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
    f"Output classes: {NUM_CLASSES}"
)

print(
    "Training dataset: Rabbani Bleed Seg"
)

print(
    "Evaluation dataset: HemoSet test split"
)

print(
    f"Checkpoint: {checkpoint_path}"
)

print(
    f"Test CSV: {config_split.CSV_TEST_PATH}"
)

print(
    f"Test images: {len(test_ds)}"
)


# ============================================================
# INFERENCE
# ============================================================

tp_batches = []
fp_batches = []
fn_batches = []
tn_batches = []


with torch.no_grad():

    for batch_index, (
        test_img,
        test_mask,
    ) in enumerate(test_hemo_dl):

        test_img = test_img.to(
            "cuda",
            non_blocking=True,
        )

        test_mask = prepare_mask(
            test_mask
        )

        logits = model(
            test_img
        )

        predictions = get_predictions(
            logits
        )

        batch_tp, batch_fp, batch_fn, batch_tn = (
            get_segmentation_stats(
                predictions,
                test_mask,
            )
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
                f"{batch_index}/{len(test_hemo_dl)}"
            )


# Join all batches while keeping every image separate.
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


# Select only the blood class.
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
    correction=0,
).item()


mean_dice = dice_per_image.mean().item()

std_dice = dice_per_image.std(
    correction=0,
).item()


mean_precision = precision_per_image.mean().item()

std_precision = precision_per_image.std(
    correction=0,
).item()


mean_recall = recall_per_image.mean().item()

std_recall = recall_per_image.std(
    correction=0,
).item()


# ============================================================
# DATASET-LEVEL METRICS
# ============================================================

# Sum the confusion statistics across all HemoSet test images while
# preserving the class dimension.
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


print(
    "\n----- Test image composition -----"
)

print(
    f"Total images: {total_images}"
)

print(
    f"Images with blood: {images_with_blood}"
)

print(
    f"Images without blood: {empty_images}"
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
# PER-VIDEO HEMOSET METRICS
# ============================================================

if config_split.VIDEOID_STRING not in test_ds.csv_dirs.columns:
    raise KeyError(
        "Video ID column not found in the HemoSet test CSV: "
        f"{config_split.VIDEOID_STRING}"
    )


test_video_ids = test_ds.csv_dirs[
    config_split.VIDEOID_STRING
].astype(str)


print(
    "\n----- HemoSet test images per video -----"
)

print(
    test_video_ids.value_counts()
)


print(
    "\n----- Per-video blood metrics -----"
)


# Use the order defined for the HemoSet test split.
for video_id in config_split.TEST_VIDEO_ID:

    video_mask = torch.tensor(
        (
            test_video_ids == video_id
        ).to_numpy(),
        dtype=torch.bool,
    )

    video_image_count = video_mask.sum().item()

    if video_image_count == 0:
        print(
            f"\nVideo: {video_id}"
        )

        print(
            "No images found in the test CSV."
        )

        continue


    # Per-image metrics for the current video.
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


    # Dataset-level confusion statistics for the current video.
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

    video_global_precision = video_precision_classes[
        BLOOD_CLASS_INDEX
    ].item()

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
        f"\nVideo: {video_id}"
    )

    print(
        f"Images: {video_image_count}"
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
        f"Global IoU:       {video_global_iou:.4f}"
    )

    print(
        f"Global Dice:      {video_global_dice:.4f}"
    )

    print(
        f"Global Precision: {video_global_precision:.4f}"
    )

    print(
        f"Global Recall:    {video_global_recall:.4f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n======== RABBANI TO HEMOSET ZERO-SHOT RESULTS ========"
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
    "Training dataset: Rabbani Bleed Seg"
)

print(
    "Test dataset: HemoSet"
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