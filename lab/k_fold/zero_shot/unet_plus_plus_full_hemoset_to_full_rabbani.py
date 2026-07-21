"""
file: unet_plus_plus_full_hemoset_to_full_rabbani.py

brief:  this script trains U-Net++ with a ResNet-18 encoder on the complete
        HemoSet dataset divided into train and validation, then
        performs zero-shot evaluation on the complete Rabbani
        full_labeled_dataset.csv file.

    Positional argument: number of epochs. Default: 50.
"""

import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from src.data_transforms import (
    create_train_transform,
    create_eval_transform,
    create_bleed_train_transform,
    create_bleed_eval_transform,
)
from src.hemoset_dataset_v2 import CustomImageDataset
from src import config_split
from torch.utils.data import DataLoader


# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings on the CPU node.
torch.backends.nnpack.set_flags(False)


device = "cuda"
batch_size = 4
num_workers = 2
encoder_name = "resnet18"
learning_rate = 0.001
random_seed = 42

segmentation_mode = config_split.SEGMENTATION_MODE.strip().lower()
output_classes = 1 if segmentation_mode == "binary" else config_split.NUM_CLASSES


"""
function: set_random_seed
brief:    this routine sets all random seeds before model training
"""
def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


"""
function: prepare_mask
brief:    this routine prepares masks for loss and metric computation
"""
def prepare_mask(mask, mode):
    if mode == "binary":
        return (mask > 0).float().to(device)
    return torch.squeeze(mask, 1).long().to(device)


"""
function: compute_loss
brief:    this routine computes the selected segmentation loss
"""
def compute_loss(mode, logits, mask, bce_loss, dice_loss, ce_loss):
    if mode == "binary":
        return bce_loss(logits, mask) + dice_loss(logits, mask)
    return ce_loss(logits, mask)


"""
function: get_predictions
brief:    this routine converts model logits into predicted masks
"""
def get_predictions(logits, mode):
    if mode == "binary":
        return (torch.sigmoid(logits) >= config_split.BINARY_THRESHOLD).long()
    return logits.argmax(dim=1)


"""
function: get_segmentation_stats
brief:    this routine retrieves TP, FP, FN and TN for every image
"""
def get_segmentation_stats(predictions, mask, mode):
    if mode == "binary":
        return smp.metrics.get_stats(predictions, mask.long(), mode=mode)
    return smp.metrics.get_stats(
        predictions,
        mask,
        mode=mode,
        num_classes=config_split.NUM_CLASSES,
    )


"""
function: select_blood_class
brief:    this routine selects the blood class from global or per-image values
"""
def select_blood_class(values, mode, per_image=False):
    class_index = 0 if mode == "binary" else 1
    return values[:, class_index] if per_image else values[class_index]


"""
function: compute_metrics
brief:    this routine computes global and per-image segmentation metrics
"""
def compute_metrics(tp, fp, fn, tn, mode):
    image_iou = select_blood_class(
        smp.metrics.iou_score(tp, fp, fn, tn, reduction="none"),
        mode,
        per_image=True,
    )
    image_dice = select_blood_class(
        smp.metrics.f1_score(tp, fp, fn, tn, reduction="none"),
        mode,
        per_image=True,
    )
    image_precision = select_blood_class(
        smp.metrics.precision(tp, fp, fn, tn, reduction="none"),
        mode,
        per_image=True,
    )
    image_recall = select_blood_class(
        smp.metrics.recall(tp, fp, fn, tn, reduction="none"),
        mode,
        per_image=True,
    )

    global_tp = tp.sum(dim=0)
    global_fp = fp.sum(dim=0)
    global_fn = fn.sum(dim=0)
    global_tn = tn.sum(dim=0)

    global_iou = select_blood_class(
        smp.metrics.iou_score(global_tp, global_fp, global_fn, global_tn, reduction="none"),
        mode,
    )
    global_dice = select_blood_class(
        smp.metrics.f1_score(global_tp, global_fp, global_fn, global_tn, reduction="none"),
        mode,
    )
    global_precision = select_blood_class(
        smp.metrics.precision(global_tp, global_fp, global_fn, global_tn, reduction="none"),
        mode,
    )
    global_recall = select_blood_class(
        smp.metrics.recall(global_tp, global_fp, global_fn, global_tn, reduction="none"),
        mode,
    )

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
    }


"""
function: create_model
brief:    this routine creates a U-Net++ model with a ResNet-18 encoder
"""
def create_model(encoder_weights):
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=output_classes,
    )
    return model.to(device)


"""
function: save_checkpoint
brief:    this routine safely saves the best checkpoint
"""
def save_checkpoint(checkpoint, checkpoint_path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)


"""
function: load_checkpoint
brief:    this routine loads a structured checkpoint
"""
def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint format not supported: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


lab_directory = Path(__file__).resolve().parents[2]
source_train_csv = lab_directory / "k_fold" / "zero_shot" / "split_dataset" / "generated" / "hemoset" / "train.csv"
source_validation_csv = lab_directory / "k_fold" / "zero_shot" / "split_dataset" / "generated" / "hemoset" / "validation.csv"
target_full_csv = lab_directory / "splits_v1p0" / "full_labeled_dataset.csv"
SOURCE_TRAIN_TRANSFORM = create_train_transform
SOURCE_EVAL_TRANSFORM = create_eval_transform
TARGET_EVAL_TRANSFORM = create_bleed_eval_transform
SOURCE_DATASET_NAME = "hemoset"
TARGET_DATASET_NAME = "rabbani"

checkpoint_directory = (
    Path("/work/cvcs2026/latent_space_cowboys/model_pretrained")
    / "k_fold"
    / "zero_shot"
    / "unet_plus_plus_resnet18"
    / "full_hemoset_to_full_rabbani"
    / config_split.SEGMENTATION_MODE
)

if len(sys.argv) > 1 and sys.argv[1] != "":
    n_epochs = int(sys.argv[1])
else:
    n_epochs = 50
if len(sys.argv) > 2:
    raise ValueError("Usage: python -m k_fold.zero_shot.unet_plus_plus_full_hemoset_to_full_rabbani [epochs]")
if n_epochs <= 0:
    raise ValueError("The number of epochs must be greater than zero.")

for csv_path in [source_train_csv, source_validation_csv, target_full_csv]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

train_loader = DataLoader(
    CustomImageDataset(str(source_train_csv), SOURCE_TRAIN_TRANSFORM()),
    batch_size,
    num_workers=num_workers,
    shuffle=True,
)
validation_loader = DataLoader(
    CustomImageDataset(str(source_validation_csv), SOURCE_EVAL_TRANSFORM()),
    batch_size,
    num_workers=num_workers,
    shuffle=False,
)
test_loader = DataLoader(
    CustomImageDataset(str(target_full_csv), TARGET_EVAL_TRANSFORM()),
    batch_size,
    num_workers=num_workers,
    shuffle=False,
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

set_random_seed(random_seed)
print(f"number of epochs: {n_epochs}")
print(f"segmentation mode: {segmentation_mode}")
print(f"training dataset: {SOURCE_DATASET_NAME}")
print(f"zero-shot test dataset: {TARGET_DATASET_NAME}")
print(f"train images: {len(train_loader.dataset)}")
print(f"validation images: {len(validation_loader.dataset)}")
print(f"target test images: {len(test_loader.dataset)}")

train_img, train_mask = next(iter(train_loader))
print(f"Feature batch shape: {train_img.size()}")
print(f"Labels batch shape: {train_mask.size()}")

model = create_model("imagenet")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
bce_loss = torch.nn.BCEWithLogitsLoss().to(device)
dice_loss = smp.losses.DiceLoss(mode=segmentation_mode, from_logits=True).to(device)
ce_loss = torch.nn.CrossEntropyLoss().to(device)

checkpoint_path = (
    checkpoint_directory
    / f"unet_plus_plus_zero_shot_full_hemo_to_full_rab_{segmentation_mode}_best_{encoder_name}.pth"
)
best_val_loss = float("inf")
best_epoch = 0
best_metrics = None

for epoch in range(n_epochs):
    print(f"------------ EPOCH: {epoch + 1} ------------")
    model.train()
    train_loss_sum = 0.0
    for i, (train_img, train_mask) in enumerate(train_loader):
        optimizer.zero_grad()
        train_img = train_img.to(device)
        train_mask = prepare_mask(train_mask, segmentation_mode)
        logits = model(train_img)
        loss = compute_loss(segmentation_mode, logits, train_mask, bce_loss, dice_loss, ce_loss)
        if i % 50 == 0:
            print(f"loss_train {loss}")
        train_loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss_sum / len(train_loader)

    model.eval()
    val_loss_sum = 0.0
    tp_list, fp_list, fn_list, tn_list = [], [], [], []
    with torch.no_grad():
        for j, (val_img, val_mask) in enumerate(validation_loader):
            val_img = val_img.to(device)
            val_mask = prepare_mask(val_mask, segmentation_mode)
            logits = model(val_img)
            val_loss = compute_loss(segmentation_mode, logits, val_mask, bce_loss, dice_loss, ce_loss)
            if j % 50 == 0:
                print(f"loss_valid {val_loss}")
            val_loss_sum += val_loss.item()
            predictions = get_predictions(logits, segmentation_mode)
            tp, fp, fn, tn = get_segmentation_stats(predictions, val_mask, segmentation_mode)
            tp_list.append(tp.cpu())
            fp_list.append(fp.cpu())
            fn_list.append(fn.cpu())
            tn_list.append(tn.cpu())

    avg_val_loss = val_loss_sum / len(validation_loader)
    validation_metrics = compute_metrics(
        torch.cat(tp_list, dim=0),
        torch.cat(fp_list, dim=0),
        torch.cat(fn_list, dim=0),
        torch.cat(tn_list, dim=0),
        segmentation_mode,
    )
    current_lr = scheduler.get_last_lr()[0]

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        best_metrics = dict(validation_metrics)
        checkpoint = {
            "model_name": "unet_plus_plus",
            "encoder_name": encoder_name,
            "segmentation_mode": segmentation_mode,
            "output_classes": output_classes,
            "training_dataset": SOURCE_DATASET_NAME,
            "zero_shot_test_dataset": TARGET_DATASET_NAME,
            "epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "validation_metrics": best_metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, checkpoint_path)
        print(f"new best checkpoint: epoch {best_epoch}, validation loss {best_val_loss:.6f}")

    scheduler.step()
    print(f"learning_rate {current_lr}")
    print(f"----- Avg values for epoch {epoch + 1} -----")
    print(f"avg_train_loss {avg_train_loss}")
    print(f"avg_val_loss {avg_val_loss}")
    print(f"global_iou {validation_metrics['global_iou']}")
    print(f"global_dice {validation_metrics['global_dice']}")
    print(f"image_iou {validation_metrics['image_iou_mean']} +/- {validation_metrics['image_iou_std']}")
    print(f"image_dice {validation_metrics['image_dice_mean']} +/- {validation_metrics['image_dice_std']}")

if best_metrics is None:
    raise RuntimeError("No checkpoint was saved.")

checkpoint = load_checkpoint(model, checkpoint_path)
model.eval()
tp_list, fp_list, fn_list, tn_list = [], [], [], []
with torch.no_grad():
    for i, (test_img, test_mask) in enumerate(test_loader):
        test_img = test_img.to(device)
        test_mask = prepare_mask(test_mask, segmentation_mode)
        logits = model(test_img)
        predictions = get_predictions(logits, segmentation_mode)
        tp, fp, fn, tn = get_segmentation_stats(predictions, test_mask, segmentation_mode)
        tp_list.append(tp.cpu())
        fp_list.append(fp.cpu())
        fn_list.append(fn.cpu())
        tn_list.append(tn.cpu())
        if i % 50 == 0:
            print(f"evaluated zero-shot batch {i}/{len(test_loader)}")

test_metrics = compute_metrics(
    torch.cat(tp_list, dim=0),
    torch.cat(fp_list, dim=0),
    torch.cat(fn_list, dim=0),
    torch.cat(tn_list, dim=0),
    segmentation_mode,
)

print("\n============================================================")
print("FINAL ZERO-SHOT RESULTS")
print("============================================================")
print(f"best_epoch {checkpoint['epoch']}")
print(f"best_validation_loss {checkpoint['best_validation_loss']}")
print(f"global_iou {test_metrics['global_iou']}")
print(f"global_dice {test_metrics['global_dice']}")
print(f"global_precision {test_metrics['global_precision']}")
print(f"global_recall {test_metrics['global_recall']}")
print(f"image_iou {test_metrics['image_iou_mean']} +/- {test_metrics['image_iou_std']}")
print(f"image_dice {test_metrics['image_dice_mean']} +/- {test_metrics['image_dice_std']}")
print(f"image_precision {test_metrics['image_precision_mean']} +/- {test_metrics['image_precision_std']}")
print(f"image_recall {test_metrics['image_recall_mean']} +/- {test_metrics['image_recall_std']}")
print(f"checkpoint {checkpoint_path}")
