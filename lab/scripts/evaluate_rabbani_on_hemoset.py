"""
file: evaluate_hemoset_on_rabbani.py

brief:
    Perform binary zero-shot evaluation on the Rabbani test split using one
    model trained on HemoSet.

    Select the architecture in src/config_split.py with:

        RABBANI_EVALUATION_MODEL = "unet_plus_plus"

    or:

        RABBANI_EVALUATION_MODEL = "deeplabv3plus"

usage:
    python -u -m scripts.rabbani.evaluate_hemoset_on_rabbani
"""

import os

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import create_bleed_eval_transform
from src.hemoset_dataset_v2 import CustomImageDataset


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = config_split.RABBANI_EVALUATION_MODEL.strip().lower()
SUPPORTED_MODELS = {"unet_plus_plus", "deeplabv3plus"}

if MODEL_NAME not in SUPPORTED_MODELS:
    raise ValueError(
        "RABBANI_EVALUATION_MODEL must be "
        "'unet_plus_plus' or 'deeplabv3plus'. "
        f"Received: {MODEL_NAME}"
    )

ENCODER_NAME = "resnet18"
NUM_OUTPUT_CHANNELS = 1
BLOOD_CLASS_INDEX = 0
BINARY_THRESHOLD = 0.50

BATCH_SIZE = 4
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DeepLabV3+ parameters must match the training script.
ENCODER_OUTPUT_STRIDE = 16
DECODER_CHANNELS = 256
DECODER_ATROUS_RATES = (12, 24, 36)
UPSAMPLING_FACTOR = 4


# ============================================================
# HARDWARE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

# Required by the Tesla K80 environment.
torch.backends.cudnn.enabled = False
torch.backends.nnpack.set_flags(False)


# ============================================================
# MODEL
# ============================================================

def create_model_and_checkpoint():
    """Create the selected model and return its checkpoint path."""
    if MODEL_NAME == "unet_plus_plus":
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_OUTPUT_CHANNELS,
            activation=None,
        )

        display_name = "U-Net++"
        checkpoint_filename = (
            "unet_plus_plus_binary_best_resnet18.pth"
        )

    else:
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

        display_name = "DeepLabV3+"
        checkpoint_filename = (
            "deeplabv3plus_binary_best_resnet18_hemo.pth"
        )

    checkpoint_path = os.path.join(
        config_split.MODEL_PRETRAINED_DIR,
        checkpoint_filename,
    )

    return model.to(DEVICE), display_name, checkpoint_path


def load_checkpoint(model, checkpoint_path):
    """Load a plain state dictionary or a structured checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if os.path.getsize(checkpoint_path) == 0:
        raise EOFError(
            f"The checkpoint file is empty: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )
    except EOFError as error:
        raise EOFError(
            "The checkpoint exists but is incomplete or corrupted: "
            f"{checkpoint_path}"
        ) from error

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()


# ============================================================
# HELPERS
# ============================================================

def prepare_mask(mask):
    """Convert masks encoded as 0/1 or 0/255 to binary float masks."""
    return (mask > 0).float().to(
        DEVICE,
        non_blocking=True,
    )


def get_predictions(logits):
    """Convert logits to binary predictions."""
    return (
        torch.sigmoid(logits) >= BINARY_THRESHOLD
    ).long()


def compute_metrics(tp, fp, fn, tn):
    """Compute IoU, Dice, precision and recall."""
    return (
        smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="none"
        ),
        smp.metrics.f1_score(
            tp, fp, fn, tn, reduction="none"
        ),
        smp.metrics.precision(
            tp, fp, fn, tn, reduction="none"
        ),
        smp.metrics.recall(
            tp, fp, fn, tn, reduction="none"
        ),
    )


# ============================================================
# DATASET
# ============================================================

eval_transform = create_bleed_eval_transform()

test_dataset = CustomImageDataset(
    config_split.CSV_TEST_PATH_V1P0,
    eval_transform,
)

if len(test_dataset) == 0:
    raise ValueError("The Rabbani test dataset is empty.")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(NUM_WORKERS > 0),
)


# ============================================================
# MODEL AND CHECKPOINT
# ============================================================

model, model_display_name, checkpoint_path = (
    create_model_and_checkpoint()
)

load_checkpoint(model, checkpoint_path)


# ============================================================
# SUMMARY
# ============================================================

print("======== HEMOSET TO RABBANI ZERO-SHOT EVALUATION ========", flush=True)
print(f"Model: {model_display_name}", flush=True)
print(f"Configuration value: {MODEL_NAME}", flush=True)
print(f"Encoder: {ENCODER_NAME}", flush=True)
print("Segmentation mode: binary", flush=True)
print("Output channels: 1", flush=True)
print("Training loss: BCEWithLogitsLoss + DiceLoss", flush=True)
print(f"Binary threshold: {BINARY_THRESHOLD:.2f}", flush=True)
print("Training dataset: HemoSet", flush=True)
print("Evaluation dataset: Rabbani test split", flush=True)
print(f"Checkpoint: {checkpoint_path}", flush=True)
print(
    "Checkpoint size: "
    f"{os.path.getsize(checkpoint_path) / (1024 ** 2):.2f} MiB",
    flush=True,
)
print(f"Test CSV: {config_split.CSV_TEST_PATH_V1P0}", flush=True)
print(f"Test images: {len(test_dataset)}", flush=True)
print(f"Test batches: {len(test_loader)}", flush=True)


# ============================================================
# INFERENCE
# ============================================================

tp_batches = []
fp_batches = []
fn_batches = []
tn_batches = []

with torch.inference_mode():
    for batch_index, (images, masks) in enumerate(test_loader):
        images = images.to(DEVICE, non_blocking=True)
        masks = prepare_mask(masks)

        logits = model(images)

        if logits.shape != masks.shape:
            raise RuntimeError(
                "Unexpected output shape. "
                f"Received {tuple(logits.shape)}, "
                f"expected {tuple(masks.shape)}."
            )

        if batch_index == 0:
            print(f"Input batch shape: {tuple(images.shape)}", flush=True)
            print(f"Output batch shape: {tuple(logits.shape)}", flush=True)
            print(f"Mask batch shape: {tuple(masks.shape)}", flush=True)

        predictions = get_predictions(logits)

        batch_tp, batch_fp, batch_fn, batch_tn = (
            smp.metrics.get_stats(
                predictions,
                masks.long(),
                mode="binary",
            )
        )

        tp_batches.append(batch_tp.cpu())
        fp_batches.append(batch_fp.cpu())
        fn_batches.append(batch_fn.cpu())
        tn_batches.append(batch_tn.cpu())

        if batch_index % 25 == 0:
            print(
                f"Evaluated batch {batch_index}/{len(test_loader)}",
                flush=True,
            )


tp = torch.cat(tp_batches, dim=0)
fp = torch.cat(fp_batches, dim=0)
fn = torch.cat(fn_batches, dim=0)
tn = torch.cat(tn_batches, dim=0)


# ============================================================
# PER-IMAGE METRICS
# ============================================================

image_iou_classes, image_dice_classes, image_precision_classes, image_recall_classes = (
    compute_metrics(tp, fp, fn, tn)
)

image_iou = image_iou_classes[:, BLOOD_CLASS_INDEX]
image_dice = image_dice_classes[:, BLOOD_CLASS_INDEX]
image_precision = image_precision_classes[:, BLOOD_CLASS_INDEX]
image_recall = image_recall_classes[:, BLOOD_CLASS_INDEX]

mean_iou = image_iou.mean().item()
std_iou = image_iou.std(unbiased=False).item()

mean_dice = image_dice.mean().item()
std_dice = image_dice.std(unbiased=False).item()

mean_precision = image_precision.mean().item()
std_precision = image_precision.std(unbiased=False).item()

mean_recall = image_recall.mean().item()
std_recall = image_recall.std(unbiased=False).item()


# ============================================================
# DATASET-LEVEL METRICS
# ============================================================

global_tp = tp.sum(dim=0)
global_fp = fp.sum(dim=0)
global_fn = fn.sum(dim=0)
global_tn = tn.sum(dim=0)

global_iou_classes, global_dice_classes, global_precision_classes, global_recall_classes = (
    compute_metrics(global_tp, global_fp, global_fn, global_tn)
)

global_iou = global_iou_classes[BLOOD_CLASS_INDEX].item()
global_dice = global_dice_classes[BLOOD_CLASS_INDEX].item()
global_precision = global_precision_classes[BLOOD_CLASS_INDEX].item()
global_recall = global_recall_classes[BLOOD_CLASS_INDEX].item()


# ============================================================
# EMPTY IMAGE STATISTICS
# ============================================================

blood_gt_pixels = (
    tp[:, BLOOD_CLASS_INDEX] + fn[:, BLOOD_CLASS_INDEX]
)

blood_pred_pixels = (
    tp[:, BLOOD_CLASS_INDEX] + fp[:, BLOOD_CLASS_INDEX]
)

images_with_blood = (blood_gt_pixels > 0).sum().item()
empty_images = (blood_gt_pixels == 0).sum().item()

correct_empty_predictions = (
    (blood_gt_pixels == 0) & (blood_pred_pixels == 0)
).sum().item()

empty_false_positives = (
    (blood_gt_pixels == 0) & (blood_pred_pixels > 0)
).sum().item()


# ============================================================
# RESULTS
# ============================================================

print("\n======== ZERO-SHOT RESULTS ========", flush=True)
print(f"Model: {model_display_name}", flush=True)
print(f"Checkpoint: {checkpoint_path}", flush=True)
print(f"Test images: {len(test_dataset)}", flush=True)

print("\n----- Test image composition -----", flush=True)
print(f"Images with blood: {images_with_blood}", flush=True)
print(f"Images without blood: {empty_images}", flush=True)
print(
    f"Correctly predicted as empty: {correct_empty_predictions}",
    flush=True,
)
print(
    "Empty images with false-positive blood: "
    f"{empty_false_positives}",
    flush=True,
)

print("\n----- Dataset-level blood metrics -----", flush=True)
print(f"IoU:       {global_iou:.4f}", flush=True)
print(f"Dice:      {global_dice:.4f}", flush=True)
print(f"Precision: {global_precision:.4f}", flush=True)
print(f"Recall:    {global_recall:.4f}", flush=True)

print("\n----- Per-image blood metrics -----", flush=True)
print(f"IoU:       {mean_iou:.4f} +/- {std_iou:.4f}", flush=True)
print(f"Dice:      {mean_dice:.4f} +/- {std_dice:.4f}", flush=True)
print(
    f"Precision: {mean_precision:.4f} +/- {std_precision:.4f}",
    flush=True,
)
print(f"Recall:    {mean_recall:.4f} +/- {std_recall:.4f}", flush=True)