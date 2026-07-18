"""
file: evaluate_final_hemoset_on_rabbani.py

brief:
    This script evaluates the final segmentation model trained on the complete
    HemoSet development set using the Rabbani Bleed Seg test split.

    The model architecture, encoder, segmentation mode and final number of
    training epochs are read from config_split, matching the configuration
    used by final_train.py.

    No Rabbani image is used during HemoSet training. Therefore, this script
    performs a zero-shot cross-dataset evaluation:

        training dataset: HemoSet
        test dataset: Rabbani Bleed Seg

    Dataset-level metrics are computed by summing TP, FP, FN and TN across all
    test images before computing IoU, Dice, precision and recall.

    Per-image metrics are computed independently for every test image and are
    reported using mean and standard deviation.

usage:
    python -m scripts.rabbani.evaluate_final_hemoset_on_rabbani
"""

import os

import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import create_bleed_eval_transform
from src.hemoset_dataset_v2 import CustomImageDataset


# ============================================================
# FINAL HEMOSET MODEL CONFIGURATION
# ============================================================

MODEL_NAME = config_split.FINAL_MODEL_NAME
ENCODER_NAME = config_split.FINAL_ENCODER_NAME

SEGMENTATION_MODE = config_split.SEGMENTATION_MODE
NUM_CLASSES = config_split.NUM_CLASSES

BATCH_SIZE = config_split.FINAL_BATCH_SIZE
NUM_WORKERS = config_split.FINAL_NUM_WORKERS


# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings on the CPU node.
torch.backends.nnpack.set_flags(False)


def resolve_final_num_epochs():
    """
    Read the number of final training epochs from config_split.

    The function supports both possible configuration structures:

    FINAL_NUM_EPOCHS[model][encoder]

    and:

    FINAL_NUM_EPOCHS[segmentation_mode][model][encoder]
    """
    epoch_configuration = config_split.FINAL_NUM_EPOCHS

    if SEGMENTATION_MODE in epoch_configuration:
        return epoch_configuration[
            SEGMENTATION_MODE
        ][
            MODEL_NAME
        ][
            ENCODER_NAME
        ]

    return epoch_configuration[
        MODEL_NAME
    ][
        ENCODER_NAME
    ]


def create_model():
    """
    Create the same architecture used during final HemoSet training.
    """
    if MODEL_NAME == "unet":
        return smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        ).to("cuda")

    if MODEL_NAME == "unet_plus_plus":
        return smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        ).to("cuda")

    raise ValueError(
        f"Unsupported final model: {MODEL_NAME}"
    )


def prepare_mask(mask):
    """
    Prepare the target mask according to the segmentation mode.
    """
    if SEGMENTATION_MODE == "binary":
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


def get_predictions(logits):
    """
    Convert model logits into the final predicted segmentation map.
    """
    if SEGMENTATION_MODE == "binary":
        return (
            torch.sigmoid(logits)
            >= config_split.BINARY_THRESHOLD
        ).long()

    return logits.argmax(
        dim=1,
    )


def get_segmentation_stats(predictions, mask):
    """
    Compute TP, FP, FN and TN independently for every image and class.
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
        num_classes=NUM_CLASSES,
    )


def get_blood_class_index():
    """
    Return the index corresponding to the blood class.
    """
    if SEGMENTATION_MODE == "binary":
        return 0

    return 1


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
# FINAL HEMOSET CHECKPOINT
# ============================================================

final_num_epochs = resolve_final_num_epochs()


checkpoint_path = os.path.join(
    config_split.MODEL_PRETRAINED_DIR,
    (
        f"{MODEL_NAME}_"
        f"{SEGMENTATION_MODE}_"
        f"final_"
        f"{ENCODER_NAME}_"
        f"epoch_{final_num_epochs}.pth"
    ),
)


if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
        "Final HemoSet checkpoint not found: "
        f"{checkpoint_path}"
    )


model = create_model()


model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location="cuda",
    )
)


model.eval()


# ============================================================
# RABBANI TEST DATASET
# ============================================================

# Rabbani images are resized to the resolution expected by the model.
eval_transform = create_bleed_eval_transform()


test_ds = CustomImageDataset(
    config_split.CSV_TEST_PATH_V1P0,
    eval_transform,
)


if len(test_ds) == 0:
    raise ValueError(
        "The Rabbani test dataset is empty: "
        f"{config_split.CSV_TEST_PATH_V1P0}"
    )


test_bleed_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
)


print(
    "======== FINAL HEMOSET MODEL ON RABBANI ========"
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
    f"Final HemoSet training epochs: {final_num_epochs}"
)

if SEGMENTATION_MODE == "binary":
    print(
        f"Binary threshold: "
        f"{config_split.BINARY_THRESHOLD}"
    )

print(
    "Training dataset: HemoSet"
)

print(
    "Evaluation dataset: Rabbani Bleed Seg test split"
)

print(
    f"Checkpoint: {checkpoint_path}"
)

print(
    f"Test CSV: {config_split.CSV_TEST_PATH_V1P0}"
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
    ) in enumerate(test_bleed_dl):

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
                f"{batch_index}/{len(test_bleed_dl)}"
            )


# Join all batches while keeping every test image separate.
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


blood_class_index = get_blood_class_index()


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
    blood_class_index,
]

dice_per_image = dice_classes[
    :,
    blood_class_index,
]

precision_per_image = precision_classes[
    :,
    blood_class_index,
]

recall_per_image = recall_classes[
    :,
    blood_class_index,
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

# Sum confusion statistics across all Rabbani test images.
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
    blood_class_index
].item()

global_dice = global_dice_classes[
    blood_class_index
].item()

global_precision = global_precision_classes[
    blood_class_index
].item()

global_recall = global_recall_classes[
    blood_class_index
].item()


# ============================================================
# TEST SET COMPOSITION
# ============================================================

blood_gt_pixels = (
    tp[:, blood_class_index]
    + fn[:, blood_class_index]
)

blood_pred_pixels = (
    tp[:, blood_class_index]
    + fp[:, blood_class_index]
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
# DATASET IDENTIFIER COMPOSITION
# ============================================================

if config_split.VIDEOID_STRING in test_ds.csv_dirs.columns:

    test_identifiers = test_ds.csv_dirs[
        config_split.VIDEOID_STRING
    ].astype(str)

    print(
        "\n----- Test images per dataset identifier -----"
    )

    print(
        test_identifiers.value_counts()
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n======== HEMOSET TO RABBANI ZERO-SHOT RESULTS ========"
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
    f"Final HemoSet training epochs: {final_num_epochs}"
)

print(
    "Training dataset: HemoSet"
)

print(
    "Test dataset: Rabbani Bleed Seg"
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