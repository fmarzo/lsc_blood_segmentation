"""
file: evaluate_deeplabv3plus_all_config.py

brief:  this script evaluates the best DeepLabV3+ ResNet-18 checkpoint
        associated with every selected stratified Rabbani configuration.

    The script does not train the model and does not use the validation set to
    choose a checkpoint. For each configuration, it loads the checkpoint that
    was previously selected by train_deeplabv3plus_all_config.py using the lowest validation
    loss.

    The matching config_XXX/test.csv file is loaded and evaluated exactly once.

    The model uses the same DeepLabV3+ decoder, Atrous Spatial Pyramid Pooling module and
    ResNet-18 encoder architecture used during training. The decoder also
    combines high-level features with low-level encoder features to improve
    boundary reconstruction.

    The script computes the following blood-segmentation metrics:

    - global IoU, Dice, precision and recall, calculated after summing TP, FP,
      FN and TN over every test image;

    - mean and standard deviation of per-image IoU, Dice, precision and recall.

    For every configuration, the script prints the checkpoint epoch, the
    validation loss associated with that checkpoint and all test metrics.

    After evaluating all requested configurations, it prints the mean, sample
    variance and sample standard deviation of global IoU, global Dice, mean
    per-image IoU and mean per-image Dice across the configurations.

    All results are written only to the execution log. This script does not
    create an evaluation-results directory and does not save CSV or JSON files.

    Positional argument:

        first argument: number of configurations to evaluate

    When no argument is provided, the script evaluates every available
    config_XXX directory.

"""


import gc
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from src.data_transforms import create_bleed_eval_transform
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
# lab/k_fold/rabbani/evaluation/evaluate_deeplabv3plus_all_config.py
lab_directory = Path(__file__).resolve().parents[3]

splits_directory = (
    lab_directory
    / "k_fold"
    / "rabbani"
    / "generated_splits"
)

checkpoint_directory = (
    Path("/work/cvcs2026/latent_space_cowboys/model_pretrained")
    / "k_fold"
    / "rabbani"
    / "deeplabv3plus_resnet18"
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
        "Usage: python -m k_fold.rabbani.evaluation.evaluate_deeplabv3plus_all_config "
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
function: get_configuration_directories
brief:    this routine retrieves and sorts the configuration directories
"""
def get_configuration_directories():
    if not splits_directory.is_dir():
        raise FileNotFoundError(f"Split directory not found: {splits_directory}")

    configuration_directories = [
        path
        for path in splits_directory.glob("config_*")
        if path.is_dir()
    ]

    configuration_directories.sort(key=get_configuration_number)

    if not configuration_directories:
        raise RuntimeError("No config_XXX directories were found.")

    if n_configurations is not None:
        if n_configurations > len(configuration_directories):
            raise ValueError(
                f"Requested {n_configurations} configurations, "
                f"but only {len(configuration_directories)} are available."
            )

        configuration_directories = configuration_directories[:n_configurations]

    return configuration_directories


"""
function: create_test_loader
brief:    this routine creates the DataLoader for one test configuration
"""
def create_test_loader(configuration_directory):
    test_csv_path = configuration_directory / "test.csv"

    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    eval_transform = create_bleed_eval_transform()
    test_ds = CustomImageDataset(str(test_csv_path), eval_transform)

    test_rabbani_DL = DataLoader(
        test_ds,
        batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return test_rabbani_DL


"""
function: create_model
brief:    this routine creates the DeepLabV3+ architecture used by the checkpoint
"""
def create_model():
    deeplabv3plus = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=output_classes,
    )

    return deeplabv3plus.to(device)


"""
function: load_checkpoint
brief:    this routine loads the best checkpoint for one configuration
"""
def load_checkpoint(deeplabv3plus, checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.stat().st_size == 0:
        raise EOFError(f"Checkpoint is empty: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint format not supported: {checkpoint_path}")

    deeplabv3plus.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


# ============================================================
# EVALUATE ALL CONFIGURATIONS
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

configuration_directories = get_configuration_directories()

print(f"number of configurations: {len(configuration_directories)}")
print(f"segmentation mode: {segmentation_mode}")

all_test_results = []

for configuration_directory in configuration_directories:
    configuration_id = get_configuration_number(configuration_directory)

    print("")
    print("============================================================")
    print(f"TEST CONFIGURATION: {configuration_id:03d}")
    print("============================================================")

    checkpoint_path = (
        checkpoint_directory
        / f"deeplabv3plus_rab_{segmentation_mode}_best_{encoder_name}_config_{configuration_id:03d}.pth"
    )

    test_rabbani_DL = create_test_loader(configuration_directory)

    print(f"test images: {len(test_rabbani_DL.dataset)}")

    deeplabv3plus = create_model()
    checkpoint = load_checkpoint(deeplabv3plus, checkpoint_path)

    best_epoch = int(checkpoint["epoch"])
    best_validation_loss = float(checkpoint["best_validation_loss"])

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    deeplabv3plus.eval()

    tp_list = []
    fp_list = []
    fn_list = []
    tn_list = []

    # Test evaluation does not update model parameters.
    with torch.no_grad():
        for i, (test_img, test_mask) in enumerate(test_rabbani_DL):
            test_mask = prepare_mask(test_mask, segmentation_mode)
            test_img = test_img.to(device)

            # Forward pass on the test batch.
            img_forward = deeplabv3plus(test_img)

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
                print(f"evaluated batch {i}/{len(test_rabbani_DL)}")

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
        "test_images": len(test_rabbani_DL.dataset),
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

    del test_rabbani_DL
    del deeplabv3plus
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
print("FINAL TEST RESULTS")
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