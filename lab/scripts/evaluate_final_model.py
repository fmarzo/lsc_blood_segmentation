"""
file: evaluate_final_model.py

brief:
    This script loads the final checkpoint produced by final_train.py and
    evaluates it on the HemoSet test set.

    The model architecture, encoder, segmentation mode and selected number of
    final training epochs are read from config_split.py.

    The test set is evaluated without data augmentation and without gradient
    computation.

    Dataset-level, per-image and per-video IoU and Dice scores are reported for
    the blood class.
"""

import os

import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import create_eval_transform
from src.hemoset_dataset_v2 import CustomImageDataset


# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings on the CPU node.
torch.backends.nnpack.set_flags(False)


MODEL_ALIASES = {
    "unet": "unet",
    "unet_plus_plus": "unet_plus_plus",
    "unetplusplus": "unet_plus_plus",
    "unet++": "unet_plus_plus",
}


def normalize_model_name(model_name):
    """
    Convert the configured model name to the canonical checkpoint name.
    """
    normalized_name = model_name.strip().lower()

    if normalized_name not in MODEL_ALIASES:
        supported_names = ", ".join(sorted(MODEL_ALIASES.keys()))

        raise ValueError(
            f"Unsupported FINAL_MODEL_NAME: {model_name}. "
            f"Supported values: {supported_names}"
        )

    return MODEL_ALIASES[normalized_name]


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
            f"model='{model_name}' and encoder='{encoder_name}' in "
            "config_split.FINAL_NUM_EPOCHS."
        ) from error

    if not isinstance(n_epochs, int) or n_epochs <= 0:
        raise ValueError(
            "The selected final number of epochs must be a positive integer. "
            f"Received: {n_epochs}"
        )

    return n_epochs


def create_model(model_name, encoder_name):
    """
    Create the selected model and return its final checkpoint path.
    """
    n_epochs = get_final_num_epochs(
        model_name,
        encoder_name,
    )

    model_arguments = {
        "encoder_name": encoder_name,
        "encoder_weights": None,
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

    return model.to("cuda"), checkpoint_path, n_epochs


def prepare_mask(mask, mode):
    """
    Prepare the mask for binary or multiclass evaluation.
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


def get_predictions(logits, mode):
    """
    Convert model logits into the final segmentation map.
    """
    if mode == "binary":
        return (
            torch.sigmoid(logits)
            >= config_split.BINARY_THRESHOLD
        ).long()

    return logits.argmax(
        dim=1
    )


def get_segmentation_stats(predictions, mask, mode):
    """
    Compute TP, FP, FN and TN for every test image.
    """
    if mode == "binary":
        return smp.metrics.get_stats(
            predictions,
            mask.long(),
            mode="binary",
        )

    return smp.metrics.get_stats(
        predictions,
        mask,
        mode="multiclass",
        num_classes=config_split.NUM_CLASSES,
    )


def get_blood_class_index(mode):
    """
    Return the output class index corresponding to blood.
    """
    if mode == "binary":
        return 0

    return 1


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured for GPU evaluation."
    )


validate_segmentation_configuration()


model_name = normalize_model_name(
    config_split.FINAL_MODEL_NAME
)

encoder_name = (
    config_split.FINAL_ENCODER_NAME
    .strip()
    .lower()
)


model, checkpoint_path, n_epochs = create_model(
    model_name,
    encoder_name,
)


if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
        "Final checkpoint not found: "
        f"{checkpoint_path}"
    )


checkpoint = torch.load(
    checkpoint_path,
    map_location="cuda",
)

model.load_state_dict(
    checkpoint
)

model.eval()


eval_transform = create_eval_transform()


test_ds = CustomImageDataset(
    config_split.CSV_TEST_PATH,
    eval_transform,
)


test_hemo_dl = DataLoader(
    test_ds,
    batch_size=config_split.FINAL_BATCH_SIZE,
    num_workers=config_split.FINAL_NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
)


if len(test_hemo_dl) == 0:
    raise ValueError(
        "The test DataLoader is empty."
    )


print("======== FINAL MODEL EVALUATION ========")
print(f"Model: {model_name}")
print(f"Encoder: {encoder_name}")
print(f"Segmentation mode: {config_split.SEGMENTATION_MODE}")
print(f"Output classes: {config_split.NUM_CLASSES}")
print(f"Final training epochs: {n_epochs}")
print(f"Binary threshold: {config_split.BINARY_THRESHOLD}")
print(f"Checkpoint: {checkpoint_path}")
print(f"Test CSV: {config_split.CSV_TEST_PATH}")


tp_batches = []
fp_batches = []
fn_batches = []
tn_batches = []


with torch.no_grad():

    for test_img, test_mask in test_hemo_dl:

        test_img = test_img.to(
            "cuda",
            non_blocking=True,
        )

        test_mask = prepare_mask(
            test_mask,
            config_split.SEGMENTATION_MODE,
        )

        logits = model(
            test_img
        )

        predictions = get_predictions(
            logits,
            config_split.SEGMENTATION_MODE,
        )

        batch_tp, batch_fp, batch_fn, batch_tn = (
            get_segmentation_stats(
                predictions,
                test_mask,
                config_split.SEGMENTATION_MODE,
            )
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


blood_class_index = get_blood_class_index(
    config_split.SEGMENTATION_MODE
)


# Compute one score for each image and class.
iou_classes = smp.metrics.iou_score(
    tp,
    fp,
    fn,
    tn,
    reduction="none",
)

dice_classes = smp.metrics.f1_score(
    tp,
    fp,
    fn,
    tn,
    reduction="none",
)


iou_per_image = iou_classes[
    :,
    blood_class_index,
]

dice_per_image = dice_classes[
    :,
    blood_class_index,
]


# Compute dataset-level scores by summing confusion statistics first.
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


global_iou_classes = smp.metrics.iou_score(
    global_tp,
    global_fp,
    global_fn,
    global_tn,
    reduction="none",
)

global_dice_classes = smp.metrics.f1_score(
    global_tp,
    global_fp,
    global_fn,
    global_tn,
    reduction="none",
)


global_iou = global_iou_classes[
    blood_class_index
].item()

global_dice = global_dice_classes[
    blood_class_index
].item()


# Read the video identifier associated with every test image.
test_video_ids = (
    test_ds.csv_dirs[
        config_split.CSV_VIDEO_ID_COLUMN
    ]
    .astype(str)
    .reset_index(drop=True)
)


print("\n----- Test images per video -----")
print(
    test_video_ids.value_counts()
)


# Count ground-truth and predicted blood pixels for every image.
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


print("\n----- Test image composition -----")
print(f"Total images: {total_images}")
print(f"Images with blood: {images_with_blood}")
print(f"Images without blood: {empty_images}")


print("\n----- Empty image predictions -----")
print(
    "Correctly predicted as empty: "
    f"{correct_empty_predictions}"
)

print(
    "Empty images with false-positive blood: "
    f"{empty_images_with_false_positives}"
)


mean_iou = iou_per_image.mean().item()
mean_dice = dice_per_image.mean().item()

std_iou = iou_per_image.std(
    correction=0
).item()

std_dice = dice_per_image.std(
    correction=0
).item()


print("\n----- Per-video metrics -----")


for video_id in config_split.TEST_VIDEO_ID:

    video_id_string = str(
        video_id
    )

    video_mask = torch.tensor(
        (
            test_video_ids
            == video_id_string
        ).to_numpy(),
        dtype=torch.bool,
    )

    if not video_mask.any():
        print(
            f"\nVideo: {video_id_string}"
        )

        print(
            "No test images found."
        )

        continue

    video_iou = iou_per_image[
        video_mask
    ]

    video_dice = dice_per_image[
        video_mask
    ]


    video_tp = tp[
        video_mask
    ].sum(dim=0)

    video_fp = fp[
        video_mask
    ].sum(dim=0)

    video_fn = fn[
        video_mask
    ].sum(dim=0)

    video_tn = tn[
        video_mask
    ].sum(dim=0)


    video_global_iou_classes = smp.metrics.iou_score(
        video_tp,
        video_fp,
        video_fn,
        video_tn,
        reduction="none",
    )

    video_global_dice_classes = smp.metrics.f1_score(
        video_tp,
        video_fp,
        video_fn,
        video_tn,
        reduction="none",
    )


    video_global_iou = (
        video_global_iou_classes[
            blood_class_index
        ].item()
    )

    video_global_dice = (
        video_global_dice_classes[
            blood_class_index
        ].item()
    )


    video_mean_iou = (
        video_iou.mean().item()
    )

    video_std_iou = (
        video_iou.std(
            correction=0
        ).item()
    )

    video_mean_dice = (
        video_dice.mean().item()
    )

    video_std_dice = (
        video_dice.std(
            correction=0
        ).item()
    )


    print(
        f"\nVideo: {video_id_string}"
    )

    print(
        f"Images: {video_iou.numel()}"
    )

    print(
        f"IoU:  "
        f"{video_mean_iou:.4f} "
        f"+/- {video_std_iou:.4f}"
    )

    print(
        f"Dice: "
        f"{video_mean_dice:.4f} "
        f"+/- {video_std_dice:.4f}"
    )

    print(
        f"Global IoU:  "
        f"{video_global_iou:.4f}"
    )

    print(
        f"Global Dice: "
        f"{video_global_dice:.4f}"
    )


print("\n======== FINAL RESULTS ========")

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
    f"Final training epochs: {n_epochs}"
)

print(
    f"Checkpoint: {checkpoint_path}"
)

print(
    f"Test images: {total_images}"
)


print("\n----- Dataset-level metrics -----")

print(
    f"IoU:  {global_iou:.4f}"
)

print(
    f"Dice: {global_dice:.4f}"
)


print("\n----- Per-image metrics -----")

print(
    f"IoU:  "
    f"{mean_iou:.4f} "
    f"+/- {std_iou:.4f}"
)

print(
    f"Dice: "
    f"{mean_dice:.4f} "
    f"+/- {std_dice:.4f}"
)