"""
file: evaluate_rabbani_on_hemoset.py

brief:
    Perform zero-shot evaluation on the Rabbani Bleeding Segmentation test
    split using two binary segmentation models trained on HemoSet.

    The evaluated models are fixed:

    1. DeepLabV3+ with a ResNet-18 encoder.
    2. U-Net++ with a ResNet-18 encoder.

    Both models always use binary segmentation:

    - segmentation mode: binary
    - output channels: 1
    - foreground class: blood
    - blood channel index: 0
    - prediction: sigmoid followed by a 0.50 threshold

    The models were trained using:

        BCEWithLogitsLoss + DiceLoss

    The evaluated checkpoints are fixed:

        deeplabv3plus_binary_best_resnet18_hemo.pth

        unet_plus_plus_binary_best_resnet18.pth

    This is a zero-shot cross-dataset evaluation:

        training dataset: HemoSet
        evaluation dataset: Rabbani Bleeding Segmentation v1.0

    For each model, the script reports:

    - dataset-level IoU, Dice, precision and recall
    - mean and standard deviation of per-image metrics
    - number of correctly predicted empty images
    - number of empty images containing false-positive blood predictions

    The two models are evaluated sequentially using the same Rabbani test
    split and evaluation preprocessing.

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
# FIXED SEGMENTATION CONFIGURATION
# ============================================================

SEGMENTATION_MODE = "binary"

NUM_OUTPUT_CHANNELS = 1
BLOOD_CLASS_INDEX = 0

BINARY_THRESHOLD = 0.50

TRAINING_LOSS_NAME = (
    "BCEWithLogitsLoss + DiceLoss"
)


# ============================================================
# FIXED MODEL CONFIGURATION
# ============================================================

ENCODER_NAME = "resnet18"

ENCODER_OUTPUT_STRIDE = 16
DECODER_CHANNELS = 256
DECODER_ATROUS_RATES = (12, 24, 36)
UPSAMPLING_FACTOR = 4


MODEL_SPECIFICATIONS = [
    {
        "name": "deeplabv3plus",
        "display_name": "DeepLabV3+",
        "checkpoint_filename": (
            "deeplabv3plus_binary_"
            "best_resnet18_hemo.pth"
        ),
    },
    {
        "name": "unet_plus_plus",
        "display_name": "U-Net++",
        "checkpoint_filename": (
            "unet_plus_plus_binary_"
            "best_resnet18.pth"
        ),
    },
]


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

def create_model(model_name):
    """
    Create one of the two fixed binary segmentation models.
    """
    if model_name == "deeplabv3plus":

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

    if model_name == "unet_plus_plus":

        return smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_OUTPUT_CHANNELS,
            activation=None,
        ).to(
            DEVICE
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


def get_checkpoint_path(
    checkpoint_filename,
):
    """
    Return the complete path of an HemoSet checkpoint.
    """
    return os.path.join(
        config_split.MODEL_PRETRAINED_DIR,
        checkpoint_filename,
    )


def load_checkpoint(
    model,
    checkpoint_path,
):
    """
    Load a plain state dictionary or a structured checkpoint.
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
    Prepare a binary mask while preserving [B, 1, H, W].

    Binary metrics expect the mask to contain zero and one values.
    """
    mask = mask.float().to(
        DEVICE,
        non_blocking=True,
    )

    unique_values = torch.unique(
        mask
    )

    if not torch.all(
        (unique_values == 0)
        | (unique_values == 1)
    ):
        raise ValueError(
            "The binary target mask contains values other "
            f"than zero and one: {unique_values.tolist()}"
        )

    return mask


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


def validate_output_shape(
    logits,
    masks,
):
    """
    Verify that binary outputs and masks have the same dimensions.
    """
    expected_shape = tuple(
        masks.shape
    )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected binary model output shape. "
            f"Received {tuple(logits.shape)}, "
            f"expected {expected_shape}."
        )


def get_segmentation_stats(
    predictions,
    masks,
):
    """
    Compute TP, FP, FN and TN independently for every image.
    """
    return smp.metrics.get_stats(
        predictions,
        masks.long(),
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
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    test_loader,
):
    """
    Evaluate one binary model on the complete Rabbani test split.
    """
    tp_batches = []
    fp_batches = []
    fn_batches = []
    tn_batches = []

    model.eval()

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

    # ========================================================
    # PER-IMAGE METRICS
    # ========================================================

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

    # ========================================================
    # TEST SET COMPOSITION
    # ========================================================

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

    return {
        "total_images": total_images,
        "images_with_blood": images_with_blood,
        "empty_images": empty_images,
        "correct_empty_predictions": (
            correct_empty_predictions
        ),
        "empty_images_with_false_positives": (
            empty_images_with_false_positives
        ),
        "global_iou": global_iou,
        "global_dice": global_dice,
        "global_precision": global_precision,
        "global_recall": global_recall,
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
    }


def print_model_results(
    model_display_name,
    checkpoint_path,
    results,
):
    """
    Print the complete evaluation results for one model.
    """
    print(
        f"\n======== {model_display_name.upper()} "
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
        f"Output channels: "
        f"{NUM_OUTPUT_CHANNELS}"
    )

    print(
        f"Training loss: "
        f"{TRAINING_LOSS_NAME}"
    )

    print(
        f"Binary threshold: "
        f"{BINARY_THRESHOLD:.2f}"
    )

    print(
        "Training dataset: HemoSet"
    )

    print(
        "Test dataset: Rabbani"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Test images: "
        f"{results['total_images']}"
    )

    print(
        "\n----- Test image composition -----"
    )

    print(
        f"Images with blood: "
        f"{results['images_with_blood']}"
    )

    print(
        f"Images without blood: "
        f"{results['empty_images']}"
    )

    print(
        "\n----- Empty image predictions -----"
    )

    print(
        "Correctly predicted as empty: "
        f"{results['correct_empty_predictions']}"
    )

    print(
        "Empty images with false-positive blood: "
        f"{results['empty_images_with_false_positives']}"
    )

    print(
        "\n----- Dataset-level blood metrics -----"
    )

    print(
        f"IoU:       "
        f"{results['global_iou']:.4f}"
    )

    print(
        f"Dice:      "
        f"{results['global_dice']:.4f}"
    )

    print(
        f"Precision: "
        f"{results['global_precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{results['global_recall']:.4f}"
    )

    print(
        "\n----- Per-image blood metrics -----"
    )

    print(
        f"IoU:       "
        f"{results['mean_iou']:.4f} "
        f"+/- {results['std_iou']:.4f}"
    )

    print(
        f"Dice:      "
        f"{results['mean_dice']:.4f} "
        f"+/- {results['std_dice']:.4f}"
    )

    print(
        f"Precision: "
        f"{results['mean_precision']:.4f} "
        f"+/- {results['std_precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{results['mean_recall']:.4f} "
        f"+/- {results['std_recall']:.4f}"
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
# GLOBAL CONFIGURATION SUMMARY
# ============================================================

print(
    "======== HEMOSET TO RABBANI "
    "ZERO-SHOT EVALUATION ========"
)

print(
    "Models: DeepLabV3+ binary, U-Net++ binary"
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
    f"Training loss: "
    f"{TRAINING_LOSS_NAME}"
)

print(
    f"Binary threshold: "
    f"{BINARY_THRESHOLD:.2f}"
)

print(
    "Training dataset: HemoSet"
)

print(
    "Evaluation dataset: Rabbani test split"
)

print(
    f"Test CSV: "
    f"{config_split.CSV_TEST_PATH_V1P0}"
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
# EVALUATE BOTH FIXED MODELS
# ============================================================

all_results = {}


for model_specification in MODEL_SPECIFICATIONS:

    model_name = model_specification[
        "name"
    ]

    model_display_name = model_specification[
        "display_name"
    ]

    checkpoint_path = get_checkpoint_path(
        model_specification[
            "checkpoint_filename"
        ]
    )

    print(
        f"\n======== EVALUATING "
        f"{model_display_name.upper()} ========"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    model = create_model(
        model_name
    )

    load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
    )

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
    )

    all_results[
        model_name
    ] = results

    print_model_results(
        model_display_name=model_display_name,
        checkpoint_path=checkpoint_path,
        results=results,
    )

    del model

    torch.cuda.empty_cache()


# ============================================================
# FINAL COMPARISON
# ============================================================

print(
    "\n======== ZERO-SHOT MODEL COMPARISON ========"
)

print(
    "Training dataset: HemoSet"
)

print(
    "Test dataset: Rabbani"
)

print(
    f"Segmentation mode: "
    f"{SEGMENTATION_MODE}"
)


for model_specification in MODEL_SPECIFICATIONS:

    model_name = model_specification[
        "name"
    ]

    model_display_name = model_specification[
        "display_name"
    ]

    results = all_results[
        model_name
    ]

    print(
        f"\nModel: {model_display_name}"
    )

    print(
        f"Global IoU:       "
        f"{results['global_iou']:.4f}"
    )

    print(
        f"Global Dice:      "
        f"{results['global_dice']:.4f}"
    )

    print(
        f"Global precision: "
        f"{results['global_precision']:.4f}"
    )

    print(
        f"Global recall:    "
        f"{results['global_recall']:.4f}"
    )

    print(
        f"Mean-image Dice:  "
        f"{results['mean_dice']:.4f} "
        f"+/- {results['std_dice']:.4f}"
    )

    print(
        "Empty false positives: "
        f"{results['empty_images_with_false_positives']}"
    )