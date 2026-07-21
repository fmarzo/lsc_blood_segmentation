"""
file: unet_plus_plus_rabbani_configs_to_hemoset_configs.py

brief:  this script performs zero-shot evaluation using the best U-Net++
        ResNet-18 checkpoints previously trained on Rabbani.

    No training or validation is performed by this script.

    For every available configuration, the checkpoint selected during
    Rabbani training using the lowest validation loss is loaded and
    evaluated directly on the matching HemoSet test split:

        source checkpoint config_000 -> target config_000/test.csv
        source checkpoint config_001 -> target config_001/test.csv
        ...

    The HemoSet test images do not influence checkpoint selection and
    no model parameter is updated during evaluation.

    The script computes global IoU, Dice, precision and recall together with
    mean and standard deviation of the corresponding per-image metrics.

    After all configurations have been evaluated, it prints the mean, sample
    variance and sample standard deviation of global IoU, global Dice, mean
    per-image IoU and mean per-image Dice across configurations.

    All results are written only to the execution log.

    Positional argument:

        first argument: number of configurations to evaluate

    When no argument is provided, every configuration having both a matching
    source checkpoint and a matching target test.csv file is evaluated.
"""



import gc
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from src.data_transforms import create_eval_transform
from src.hemoset_dataset_v2 import CustomImageDataset
from src import config_split
from torch.utils.data import DataLoader


# COMMANDS USED ONLY TO RUN THE EVALUATION FROM THE BASH SCRIPT
# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings on the CPU node.
torch.backends.nnpack.set_flags(False)


# ============================================================
# GENERAL SETTINGS
# ============================================================

device = "cuda"

batch_size = 4
num_workers = 2

encoder_name = "resnet18"

# The script is located in:
# lab/k_fold/zero_shot/unet_plus_plus_rabbani_configs_to_hemoset_configs.py
lab_directory = Path(__file__).resolve().parents[2]

splits_directory = lab_directory / "k_fold" / "generated_splits"

checkpoint_directory = (
    Path("/work/cvcs2026/latent_space_cowboys/model_pretrained")
    / "k_fold"
    / "rabbani"
    / "unet_plus_plus_resnet18"
    / config_split.SEGMENTATION_MODE
)


# ============================================================
# INPUT ARGUMENT
# ============================================================

# First argument: number of configurations.
# None means all available configurations.
if len(sys.argv) > 1 and sys.argv[1] != "":
    n_configurations = int(sys.argv[1])
else:
    n_configurations = None

if len(sys.argv) > 2:
    raise ValueError(
        "Usage: python -m k_fold.zero_shot.unet_plus_plus_rabbani_configs_to_hemoset_configs "
        "[number_of_configurations]"
    )

if n_configurations is not None and n_configurations <= 0:
    raise ValueError("The number of configurations must be greater than zero.")


# ============================================================
# SEGMENTATION SETTINGS
# ============================================================

segmentation_mode = config_split.SEGMENTATION_MODE.strip().lower()

if segmentation_mode == "binary":
    output_classes = 1
else:
    output_classes = config_split.NUM_CLASSES


"""
function: prepare_mask
brief:    this routine prepares the test mask for metric computation
"""
def prepare_mask(mask, mode):
    if mode == "binary":
        return (mask > 0).float().to(device)
    else:
        return torch.squeeze(mask, 1).to(torch.long).to(device)


"""
function: get_predictions
brief:    this routine converts model logits into the final predicted mask
"""
def get_predictions(logits, mode):
    if mode == "binary":
        return (torch.sigmoid(logits) >= config_split.BINARY_THRESHOLD).long()
    else:
        return logits.argmax(dim=1)


"""
function: get_segmentation_stats
brief:    this routine retrieves TP, FP, FN and TN for every image
"""
def get_segmentation_stats(predictions, mask, mode):
    if mode == "binary":
        return smp.metrics.get_stats(predictions, mask.long(), mode=mode)
    else:
        return smp.metrics.get_stats(
            predictions,
            mask,
            mode=mode,
            num_classes=config_split.NUM_CLASSES,
        )


"""
function: get_class_target
brief:    this routine retrieves the global metric for the blood class
"""
def get_class_target(target_class, mode):
    if mode == "binary":
        return target_class[0]
    else:
        return target_class[1]


"""
function: get_image_class_target
brief:    this routine retrieves per-image metrics for the blood class
"""
def get_image_class_target(target_class, mode):
    if mode == "binary":
        return target_class[:, 0]
    else:
        return target_class[:, 1]


"""
function: compute_metrics
brief:    this routine computes global and per-image test metrics
"""
def compute_metrics(tp, fp, fn, tn, mode):
    image_iou_classes = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
    image_dice_classes = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")

    image_precision_classes = smp.metrics.precision(
        tp,
        fp,
        fn,
        tn,
        reduction="none",
    )

    image_recall_classes = smp.metrics.recall(
        tp,
        fp,
        fn,
        tn,
        reduction="none",
    )

    image_iou = get_image_class_target(image_iou_classes, mode)
    image_dice = get_image_class_target(image_dice_classes, mode)
    image_precision = get_image_class_target(image_precision_classes, mode)
    image_recall = get_image_class_target(image_recall_classes, mode)

    global_tp = tp.sum(dim=0)
    global_fp = fp.sum(dim=0)
    global_fn = fn.sum(dim=0)
    global_tn = tn.sum(dim=0)

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

    global_precision_classes = smp.metrics.precision(
        global_tp,
        global_fp,
        global_fn,
        global_tn,
        reduction="none",
    )

    global_recall_classes = smp.metrics.recall(
        global_tp,
        global_fp,
        global_fn,
        global_tn,
        reduction="none",
    )

    global_iou = get_class_target(global_iou_classes, mode)
    global_dice = get_class_target(global_dice_classes, mode)
    global_precision = get_class_target(global_precision_classes, mode)
    global_recall = get_class_target(global_recall_classes, mode)

    return {
        "global_iou": global_iou.item(),
        "global_dice": global_dice.item(),
        "global_precision": global_precision.item(),
        "global_recall": global_recall.item(),
        "image_iou_mean": image_iou.mean().item(),
        "image_iou_std": image_iou.std(unbiased=False).item(),
        "image_dice_mean": image_dice.mean().item(),
        "image_dice_std": image_dice.std(unbiased=False).item(),
        "image_precision_mean": image_precision.mean().item(),
        "image_precision_std": image_precision.std(unbiased=False).item(),
        "image_recall_mean": image_recall.mean().item(),
        "image_recall_std": image_recall.std(unbiased=False).item(),
        "tp": int(get_class_target(global_tp, mode).item()),
        "fp": int(get_class_target(global_fp, mode).item()),
        "fn": int(get_class_target(global_fn, mode).item()),
        "tn": int(get_class_target(global_tn, mode).item()),
    }


"""
function: get_configuration_number
brief:    this routine extracts the numeric identifier from config_XXX
"""
def get_configuration_number(configuration_directory):
    return int(configuration_directory.name.split("_")[-1])


"""
function: get_checkpoint_path
brief:    this routine returns the source checkpoint path for one configuration
"""
def get_checkpoint_path(configuration_id):
    return (
        checkpoint_directory
        / f"unet_plus_plus_rab_{segmentation_mode}_best_"
        f"{encoder_name}_config_{configuration_id:03d}.pth"
    )


"""
function: get_configuration_directories
brief:    this routine retrieves target configs with matching source checkpoints
"""
def get_configuration_directories():
    if not splits_directory.is_dir():
        raise FileNotFoundError(
            f"Target split directory not found: {splits_directory}"
        )

    configuration_directories = [
        path
        for path in splits_directory.glob("config_*")
        if path.is_dir()
    ]

    configuration_directories.sort(key=get_configuration_number)

    matched_configuration_directories = []

    for configuration_directory in configuration_directories:
        configuration_id = get_configuration_number(
            configuration_directory
        )

        test_csv_path = configuration_directory / "test.csv"
        checkpoint_path = get_checkpoint_path(configuration_id)

        if test_csv_path.is_file() and checkpoint_path.is_file():
            matched_configuration_directories.append(
                configuration_directory
            )

    if not matched_configuration_directories:
        raise RuntimeError(
            "No configuration has both a target test.csv "
            "and a matching source checkpoint."
        )

    if n_configurations is not None:
        if n_configurations > len(matched_configuration_directories):
            raise ValueError(
                f"Requested {n_configurations} configurations, "
                f"but only "
                f"{len(matched_configuration_directories)} "
                f"matched configurations are available."
            )

        matched_configuration_directories = (
            matched_configuration_directories[:n_configurations]
        )

    return matched_configuration_directories


"""
function: create_test_loader
brief:    this routine creates the target-domain test DataLoader
"""
def create_test_loader(configuration_directory):
    test_csv_path = configuration_directory / "test.csv"

    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    eval_transform = create_eval_transform()
    test_ds = CustomImageDataset(str(test_csv_path), eval_transform)

    test_target_DL = DataLoader(
        test_ds,
        batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return test_target_DL


"""
function: create_model
brief:    this routine creates the U-Net++ architecture used by the checkpoint
"""
def create_model():
    unet_plus_plus = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=output_classes,
    )

    return unet_plus_plus.to(device)


"""
function: load_checkpoint
brief:    this routine loads the pretrained source-domain checkpoint
"""
def load_checkpoint(unet_plus_plus, checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.stat().st_size == 0:
        raise EOFError(f"Checkpoint is empty: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint format not supported: {checkpoint_path}")

    unet_plus_plus.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


# ============================================================
# EVALUATE ALL CONFIGURATIONS
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

configuration_directories = get_configuration_directories()

print(f"number of configurations: {len(configuration_directories)}")
print(f"segmentation mode: {segmentation_mode}")
print("source dataset: Rabbani")
print("target dataset: HemoSet")
print("training performed by this script: no")

all_test_results = []

for configuration_directory in configuration_directories:
    configuration_id = get_configuration_number(configuration_directory)

    print("")
    print("============================================================")
    print(f"ZERO-SHOT CONFIGURATION: {configuration_id:03d}")
    print("============================================================")

    checkpoint_path = get_checkpoint_path(
        configuration_id
    )

    test_target_DL = create_test_loader(configuration_directory)

    print(f"test images: {len(test_target_DL.dataset)}")

    unet_plus_plus = create_model()
    checkpoint = load_checkpoint(unet_plus_plus, checkpoint_path)

    best_epoch = int(checkpoint["epoch"])
    best_validation_loss = float(checkpoint["best_validation_loss"])

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    unet_plus_plus.eval()

    tp_list = []
    fp_list = []
    fn_list = []
    tn_list = []

    # Test evaluation does not update model parameters.
    with torch.no_grad():
        for i, (test_img, test_mask) in enumerate(test_target_DL):
            test_mask = prepare_mask(test_mask, segmentation_mode)
            test_img = test_img.to(device)

            # Forward pass on the test batch.
            img_forward = unet_plus_plus(test_img)

            # Convert logits into the final segmentation masks.
            final_map = get_predictions(img_forward, segmentation_mode)

            # Collect confusion statistics for every test image.
            batch_tp, batch_fp, batch_fn, batch_tn = get_segmentation_stats(
                final_map,
                test_mask,
                segmentation_mode,
            )

            tp_list.append(batch_tp.cpu())
            fp_list.append(batch_fp.cpu())
            fn_list.append(batch_fn.cpu())
            tn_list.append(batch_tn.cpu())

            if i % 50 == 0:
                print(f"evaluated batch {i}/{len(test_target_DL)}")

    # Compute global and per-image metrics for the complete test split.
    test_metrics = compute_metrics(
        torch.cat(tp_list, dim=0),
        torch.cat(fp_list, dim=0),
        torch.cat(fn_list, dim=0),
        torch.cat(tn_list, dim=0),
        segmentation_mode,
    )

    test_result = {
        "configuration_id": configuration_id,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "checkpoint_path": str(checkpoint_path),
        "test_images": len(test_target_DL.dataset),
        **test_metrics,
    }

    all_test_results.append(test_result)

    print(f"best_epoch {best_epoch}")
    print(f"best_validation_loss {best_validation_loss}")
    print(f"global_iou {test_metrics['global_iou']}")
    print(f"global_dice {test_metrics['global_dice']}")
    print(f"global_precision {test_metrics['global_precision']}")
    print(f"global_recall {test_metrics['global_recall']}")
    print(
        f"image_iou "
        f"{test_metrics['image_iou_mean']} "
        f"+/- {test_metrics['image_iou_std']}"
    )
    print(
        f"image_dice "
        f"{test_metrics['image_dice_mean']} "
        f"+/- {test_metrics['image_dice_std']}"
    )
    print(
        f"image_precision "
        f"{test_metrics['image_precision_mean']} "
        f"+/- {test_metrics['image_precision_std']}"
    )
    print(
        f"image_recall "
        f"{test_metrics['image_recall_mean']} "
        f"+/- {test_metrics['image_recall_std']}"
    )

    del test_target_DL
    del unet_plus_plus
    del checkpoint

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# FINAL RESULTS AND STATISTICS
# ============================================================

# Metrics aggregated across all evaluated configurations.
metric_names = [
    "global_iou",
    "global_dice",
    "image_iou_mean",
    "image_dice_mean",
]

print("")
print("============================================================")
print("FINAL ZERO-SHOT RESULTS")
print("============================================================")

for result in all_test_results:
    print("")
    print(
        f"config {result['configuration_id']:03d} | "
        f"best epoch {result['best_epoch']}"
    )
    print(f"global IoU {result['global_iou']:.4f}")
    print(f"global Dice {result['global_dice']:.4f}")
    print(
        f"image IoU "
        f"{result['image_iou_mean']:.4f} "
        f"+/- {result['image_iou_std']:.4f}"
    )
    print(
        f"image Dice "
        f"{result['image_dice_mean']:.4f} "
        f"+/- {result['image_dice_std']:.4f}"
    )

print("")
print("============================================================")
print("MEAN AND VARIANCE ACROSS CONFIGURATIONS")
print("============================================================")

for metric_name in metric_names:
    # Collect the value obtained by every evaluated configuration.
    metric_values = np.array(
        [result[metric_name] for result in all_test_results],
        dtype=float,
    )

    metric_mean = np.mean(metric_values)

    # Use sample variance because the evaluated configurations represent
    # a sample of the possible grouped data splits.
    if len(metric_values) > 1:
        metric_variance = np.var(metric_values, ddof=1)
        metric_std = np.std(metric_values, ddof=1)
    else:
        metric_variance = float("nan")
        metric_std = float("nan")

    print("")
    print(f"{metric_name}")
    print(f"values {metric_values.tolist()}")
    print(f"mean {metric_mean:.4f}")
    print(f"sample_variance {metric_variance:.6f}")
    print(f"sample_standard_deviation {metric_std:.4f}")