"""
file: train_deeplabv3plus_hemoset.py

brief:  this script is the main entry point for training a DeepLabV3+ blood
    segmentation model on HemoSet.

    The model uses a ResNet-18 encoder pretrained on ImageNet. During training,
    both the pretrained encoder and the DeepLabV3+ decoder parameters are
    updated. The architecture is fixed: a DeepLabV3+ decoder with 256 channels
    on top of a ResNet-18 encoder.

    The HemoSet training split receives online augmentation to improve
    generalization, while the validation split receives only the evaluation
    preprocessing, so that validation measures performance on unaltered images.

    The segmentation approach is selected through config_split.SEGMENTATION_MODE,
    which accepts either "binary" or "multiclass". The script supports both:

    - Multiclass segmentation:
      the model produces two output channels, one for background and one for
      blood. CrossEntropyLoss compares the predicted class of every pixel with
      the corresponding ground-truth class, predictions are obtained with argmax,
      and the blood class corresponds to index 1.

    - Binary segmentation:
      the model produces one output channel representing the presence of blood.
      BCEWithLogitsLoss evaluates every pixel independently, while DiceLoss
      checks how well the entire predicted blood region overlaps the real one;
      the total loss is the sum of the two. Predictions use a sigmoid followed
      by config_split.BINARY_THRESHOLD, and the blood channel corresponds to
      index 0.

      DiceLoss is especially useful in this dataset because blood pixels can be
      much fewer than background pixels. It prevents the model from obtaining a
      good result simply by predicting mostly background and encourages it to
      correctly recover the shape and area of the blood region.

    The best checkpoint is selected using a combined score, giving equal weight
    to the dataset-level Dice and the mean per-image Dice, and its filename
    automatically records the segmentation mode that produced it.
"""

import os
import random
import sys

import numpy as np
import segmentation_models_pytorch as smp
import torch

from torch.utils.data import DataLoader

from src import config_split
from src.data_transforms import create_eval_transform, create_train_transform
from src.hemoset_dataset_v2 import CustomImageDataset

# MODEL CONFIGURATION

MODEL_NAME = "deeplabv3plus"
ENCODER_NAME = "resnet18"


SEGMENTATION_MODE = config_split.SEGMENTATION_MODE.strip().lower()


SUPPORTED_SEGMENTATION_MODES = {"binary", "multiclass"}


if SEGMENTATION_MODE not in SUPPORTED_SEGMENTATION_MODES:
    raise ValueError(
        "Unsupported SEGMENTATION_MODE value: "
        f"{SEGMENTATION_MODE}. "
        "Supported values are 'binary' and 'multiclass'."
    )


if SEGMENTATION_MODE == "binary":
    NUM_OUTPUT_CHANNELS = 1
    BLOOD_CLASS_INDEX = 0
    BINARY_THRESHOLD = getattr(config_split, "BINARY_THRESHOLD", 0.50)

else:
    NUM_OUTPUT_CHANNELS = 2
    BACKGROUND_CLASS_INDEX = 0
    BLOOD_CLASS_INDEX = 1
    BINARY_THRESHOLD = None


ENCODER_OUTPUT_STRIDE = 16
DECODER_CHANNELS = 256
DECODER_ATROUS_RATES = (12, 24, 36)
UPSAMPLING_FACTOR = 4


# TRAINING CONFIGURATION

BATCH_SIZE = 4
NUM_WORKERS = 2
ENCODER_LEARNING_RATE = 1e-4
DECODER_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_MAX_NORM = 1.0
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 12
MIN_CHECKPOINT_IMPROVEMENT = 0.001
GLOBAL_DICE_WEIGHT = 0.5
MEAN_IMAGE_DICE_WEIGHT = 0.5


# SCHEDULER CONFIGURATION

LR_REDUCTION_FACTOR = 0.5
LR_PATIENCE = 4
LR_THRESHOLD = 0.001
MINIMUM_LEARNING_RATE = 1e-6


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. " "This training script is configured for GPU execution.")


DEVICE = torch.device("cuda")


# The installed cuDNN version is not compatible with the Tesla K80.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings.
torch.backends.nnpack.set_flags(False)

torch.backends.cudnn.benchmark = False


# REPRODUCIBILITY


def configure_reproducibility(seed):
    """
    Seed Python, NumPy and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """
    Seed every DataLoader worker.
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


configure_reproducibility(RANDOM_SEED)


# MODEL HELPERS


def create_model():
    """
    Create the DeepLabV3+ ResNet-18 model.

    The number of output channels depends on SEGMENTATION_MODE.
    """
    return smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet",
        encoder_output_stride=ENCODER_OUTPUT_STRIDE,
        decoder_channels=DECODER_CHANNELS,
        decoder_atrous_rates=DECODER_ATROUS_RATES,
        in_channels=3,
        classes=NUM_OUTPUT_CHANNELS,
        activation=None,
        upsampling=UPSAMPLING_FACTOR,
    )


def freeze_encoder_batch_norm_statistics(model):
    """
    Freeze the running statistics of the pretrained encoder BatchNorm layers.

    BatchNorm affine parameters remain trainable.
    """
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


# SEGMENTATION HELPERS


def prepare_mask(mask):
    """
    Prepare masks according to SEGMENTATION_MODE.

    Binary mode:
        preserve [B, 1, H, W] and convert the mask to float.

    Multiclass mode:
        convert [B, 1, H, W] to [B, H, W] and use integer class indices.
    """
    if SEGMENTATION_MODE == "binary":
        return mask.float().to(DEVICE, non_blocking=True)
    return torch.squeeze(mask, dim=1).long().to(DEVICE, non_blocking=True)


def compute_loss(logits, mask, bce_loss, dice_loss, cross_entropy_loss):
    """
    Compute the loss associated with the selected segmentation mode.
    """
    if SEGMENTATION_MODE == "binary":
        bce_value = bce_loss(logits, mask)
        dice_value = dice_loss(logits, mask)
        return bce_value + dice_value
    return cross_entropy_loss(logits, mask)


def get_predictions(logits):
    """
    Convert logits into the final segmentation prediction.
    """
    if SEGMENTATION_MODE == "binary":
        probabilities = torch.sigmoid(logits)
        return (probabilities >= BINARY_THRESHOLD).long()
    return torch.argmax(logits, dim=1)


def get_segmentation_stats(predictions, mask):
    """
    Compute TP, FP, FN and TN independently for every image.
    """
    if SEGMENTATION_MODE == "binary":
        return smp.metrics.get_stats(predictions, mask.long(), mode="binary")
    return smp.metrics.get_stats(predictions, mask, mode="multiclass", num_classes=NUM_OUTPUT_CHANNELS)


def validate_output_shape(logits, mask):
    """
    Verify that model output and target mask shapes are compatible.
    """
    if SEGMENTATION_MODE == "binary":
        expected_shape = tuple(mask.shape)
    else:
        expected_shape = (mask.shape[0], NUM_OUTPUT_CHANNELS, mask.shape[1], mask.shape[2])
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected DeepLabV3+ output shape. "
            f"Received {tuple(logits.shape)}, "
            f"expected {expected_shape} for "
            f"{SEGMENTATION_MODE} segmentation."
        )


def compute_metrics(tp, fp, fn, tn):
    """
    Compute IoU, Dice, precision and recall.
    """
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="none")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="none")
    return (iou, dice, precision, recall)


# NUMBER OF EPOCHS

if len(sys.argv) > 1:
    n_epochs = int(sys.argv[1])

else:
    n_epochs = getattr(config_split, "DEFAULT_EPOCHS", 50)


if n_epochs <= 0:
    raise ValueError("The number of epochs must be greater than zero.")


# HEMOSET TRANSFORMS

train_transform = create_train_transform()

eval_transform = create_eval_transform()


# HEMOSET DATASETS

train_ds = CustomImageDataset(config_split.CSV_TRAIN_PATH, train_transform)


valid_ds = CustomImageDataset(config_split.CSV_VALID_PATH, eval_transform)


if len(train_ds) == 0:
    raise ValueError("The HemoSet training dataset is empty.")


if len(valid_ds) == 0:
    raise ValueError("The HemoSet validation dataset is empty.")


# DATA LOADERS

train_generator = torch.Generator()

train_generator.manual_seed(RANDOM_SEED)


validation_generator = torch.Generator()

validation_generator.manual_seed(RANDOM_SEED + 1)


train_hemo_DL = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=True,
    # Avoid a final training batch smaller than the physical batch size.
    drop_last=True,
    persistent_workers=(NUM_WORKERS > 0),
    worker_init_fn=seed_worker,
    generator=train_generator,
)


valid_hemo_DL = DataLoader(
    valid_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(NUM_WORKERS > 0),
    worker_init_fn=seed_worker,
    generator=validation_generator,
)


if len(train_hemo_DL) == 0:
    raise ValueError("The HemoSet training DataLoader is empty.")


if len(valid_hemo_DL) == 0:
    raise ValueError("The HemoSet validation DataLoader is empty.")


# DATA SHAPE CHECK

feature_batch, label_batch = next(iter(train_hemo_DL))


print(f"Feature batch shape: " f"{feature_batch.size()}")

print(f"Labels batch shape: " f"{label_batch.size()}")


# MODEL

deeplabv3plus = create_model().to(DEVICE)


# Test one batch before starting the complete training.
with torch.inference_mode():
    shape_check_images = feature_batch.to(DEVICE)
    shape_check_masks = prepare_mask(label_batch)
    shape_check_logits = deeplabv3plus(shape_check_images)
    validate_output_shape(shape_check_logits, shape_check_masks)


print(f"Model output shape: " f"{tuple(shape_check_logits.shape)}")


del feature_batch
del label_batch
del shape_check_images
del shape_check_masks
del shape_check_logits


# LOSS FUNCTIONS

if SEGMENTATION_MODE == "binary":
    bce_loss = torch.nn.BCEWithLogitsLoss().to(DEVICE)
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True).to(DEVICE)
    cross_entropy_loss = None

else:
    bce_loss = None
    dice_loss = None
    cross_entropy_loss = torch.nn.CrossEntropyLoss().to(DEVICE)


# OPTIMIZER

optimizer = torch.optim.AdamW(
    [
        {"params": (deeplabv3plus.encoder.parameters()), "lr": ENCODER_LEARNING_RATE},
        {"params": (deeplabv3plus.decoder.parameters()), "lr": DECODER_LEARNING_RATE},
        {"params": (deeplabv3plus.segmentation_head.parameters()), "lr": DECODER_LEARNING_RATE},
    ],
    weight_decay=WEIGHT_DECAY,
)


# LEARNING-RATE SCHEDULER

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=LR_REDUCTION_FACTOR,
    patience=LR_PATIENCE,
    threshold=LR_THRESHOLD,
    threshold_mode="abs",
    cooldown=1,
    min_lr=MINIMUM_LEARNING_RATE,
)


# CHECKPOINT

os.makedirs(config_split.MODEL_PRETRAINED_DIR, exist_ok=True)


checkpoint_path = os.path.join(
    config_split.MODEL_PRETRAINED_DIR, (f"deeplabv3plus_{SEGMENTATION_MODE}_" f"best_{ENCODER_NAME}_hemo.pth")
)


# TRAINING SUMMARY

print("\n======== HEMOSET DEEPLABV3+ " "TRAINING CONFIGURATION ========")

print(f"Training CSV: " f"{config_split.CSV_TRAIN_PATH}")

print(f"Validation CSV: " f"{config_split.CSV_VALID_PATH}")

print(f"Training samples: " f"{len(train_ds)}")

print(f"Validation samples: " f"{len(valid_ds)}")

print(f"Training batches: " f"{len(train_hemo_DL)}")

print(f"Validation batches: " f"{len(valid_hemo_DL)}")



if SEGMENTATION_MODE == "binary":
    print("Training loss: " "BCEWithLogitsLoss + DiceLoss")
    print(f"Binary threshold: " f"{BINARY_THRESHOLD:.2f}")

else:
    print("Training loss: CrossEntropyLoss")



# TRAINING STATE

best_selection_score = float("-inf")

best_global_dice = float("-inf")

best_mean_image_dice = float("-inf")

best_epoch = 0

epochs_without_improvement = 0


# TRAINING LOOP

for epoch in range(n_epochs):
    print(f"\n------------ EPOCH: " f"{epoch + 1}/{n_epochs} " f"------------")

    # TRAIN

    deeplabv3plus.train()
    # Keep pretrained encoder BatchNorm statistics fixed.
    freeze_encoder_batch_norm_statistics(deeplabv3plus)
    train_loss_sum = 0.0
    train_sample_count = 0
    gradient_norm_sum = 0.0
    optimizer_step_count = 0
    for batch_index, (train_images, train_masks) in enumerate(train_hemo_DL):
        optimizer.zero_grad(set_to_none=True)
        train_images = train_images.to(DEVICE, non_blocking=True)
        train_masks = prepare_mask(train_masks)
        logits = deeplabv3plus(train_images)
        if batch_index == 0:
            validate_output_shape(logits, train_masks)
        loss_train_value = compute_loss(
            logits=logits,
            mask=train_masks,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
            cross_entropy_loss=cross_entropy_loss,
        )
        if not torch.isfinite(loss_train_value):
            raise RuntimeError(
                "Non-finite training loss detected " f"at batch {batch_index}: " f"{loss_train_value.item()}"
            )
        loss_train_value.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(deeplabv3plus.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
        optimizer.step()
        current_batch_size = train_images.size(0)
        train_loss_sum += loss_train_value.item() * current_batch_size
        train_sample_count += current_batch_size
        gradient_norm_sum += float(gradient_norm.detach().cpu())
        optimizer_step_count += 1
        if batch_index % 50 == 0:
            print(f"train batch " f"{batch_index}/{len(train_hemo_DL)} " f"loss {loss_train_value.item():.6f}")
    avg_train_loss = train_loss_sum / train_sample_count
    avg_gradient_norm = gradient_norm_sum / max(optimizer_step_count, 1)

    # VALIDATION

    deeplabv3plus.eval()
    val_loss_sum = 0.0
    val_sample_count = 0
    tp_batches = []
    fp_batches = []
    fn_batches = []
    tn_batches = []
    with torch.inference_mode():
        for batch_index, (validation_images, validation_masks) in enumerate(valid_hemo_DL):
            validation_images = validation_images.to(DEVICE, non_blocking=True)
            validation_masks = prepare_mask(validation_masks)
            logits = deeplabv3plus(validation_images)
            if batch_index == 0:
                validate_output_shape(logits, validation_masks)
            loss_valid_value = compute_loss(
                logits=logits,
                mask=validation_masks,
                bce_loss=bce_loss,
                dice_loss=dice_loss,
                cross_entropy_loss=cross_entropy_loss,
            )
            if not torch.isfinite(loss_valid_value):
                raise RuntimeError(
                    "Non-finite validation loss detected " f"at batch {batch_index}: " f"{loss_valid_value.item()}"
                )
            predictions = get_predictions(logits)
            batch_tp, batch_fp, batch_fn, batch_tn = get_segmentation_stats(predictions, validation_masks)
            tp_batches.append(batch_tp.cpu())
            fp_batches.append(batch_fp.cpu())
            fn_batches.append(batch_fn.cpu())
            tn_batches.append(batch_tn.cpu())
            current_batch_size = validation_images.size(0)
            val_loss_sum += loss_valid_value.item() * current_batch_size
            val_sample_count += current_batch_size
            if batch_index % 50 == 0:
                print(f"validation batch " f"{batch_index}/{len(valid_hemo_DL)} " f"loss {loss_valid_value.item():.6f}")
    avg_val_loss = val_loss_sum / val_sample_count
    val_tp = torch.cat(tp_batches, dim=0)
    val_fp = torch.cat(fp_batches, dim=0)
    val_fn = torch.cat(fn_batches, dim=0)
    val_tn = torch.cat(tn_batches, dim=0)

    # PER-IMAGE VALIDATION METRICS

    per_image_iou_classes, per_image_dice_classes, per_image_precision_classes, per_image_recall_classes = (
        compute_metrics(val_tp, val_fp, val_fn, val_tn)
    )
    blood_iou_per_image = per_image_iou_classes[:, BLOOD_CLASS_INDEX]
    blood_dice_per_image = per_image_dice_classes[:, BLOOD_CLASS_INDEX]
    blood_precision_per_image = per_image_precision_classes[:, BLOOD_CLASS_INDEX]
    blood_recall_per_image = per_image_recall_classes[:, BLOOD_CLASS_INDEX]
    mean_image_iou = blood_iou_per_image.mean().item()
    std_image_iou = blood_iou_per_image.std(correction=0).item()
    mean_image_dice = blood_dice_per_image.mean().item()
    std_image_dice = blood_dice_per_image.std(correction=0).item()
    mean_image_precision = blood_precision_per_image.mean().item()
    std_image_precision = blood_precision_per_image.std(correction=0).item()
    mean_image_recall = blood_recall_per_image.mean().item()
    std_image_recall = blood_recall_per_image.std(correction=0).item()

    # DATASET-LEVEL VALIDATION METRICS

    global_tp = val_tp.sum(dim=0)
    global_fp = val_fp.sum(dim=0)
    global_fn = val_fn.sum(dim=0)
    global_tn = val_tn.sum(dim=0)
    global_iou_classes, global_dice_classes, global_precision_classes, global_recall_classes = compute_metrics(
        global_tp, global_fp, global_fn, global_tn
    )
    global_iou = global_iou_classes[BLOOD_CLASS_INDEX].item()
    global_dice = global_dice_classes[BLOOD_CLASS_INDEX].item()
    global_precision = global_precision_classes[BLOOD_CLASS_INDEX].item()
    global_recall = global_recall_classes[BLOOD_CLASS_INDEX].item()
    selection_score = GLOBAL_DICE_WEIGHT * global_dice + MEAN_IMAGE_DICE_WEIGHT * mean_image_dice
    if not np.isfinite(selection_score):
        raise FloatingPointError("Non-finite validation selection score detected.")
    
    # CHECKPOINT SELECTION
    
    meaningful_improvement = selection_score > best_selection_score + MIN_CHECKPOINT_IMPROVEMENT
    if meaningful_improvement:
        best_selection_score = selection_score
        best_global_dice = global_dice
        best_mean_image_dice = mean_image_dice
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(deeplabv3plus.state_dict(), checkpoint_path)
        print("\nNew best checkpoint saved.")
    else:
        epochs_without_improvement += 1
        print("\nNo meaningful validation improvement.")
        print("Epochs without improvement: " f"{epochs_without_improvement}/" f"{EARLY_STOPPING_PATIENCE}")
    
    # SCHEDULER
    
    scheduler.step(selection_score)
    encoder_current_lr = optimizer.param_groups[0]["lr"]
    decoder_current_lr = optimizer.param_groups[1]["lr"]
    
    # EPOCH SUMMARY
    
    print(f"\n----- Average values for epoch " f"{epoch + 1} -----")
    print(f"avg_train_loss: " f"{avg_train_loss:.6f}")
    print(f"avg_gradient_norm: " f"{avg_gradient_norm:.6f}")
    print(f"avg_val_loss: " f"{avg_val_loss:.6f}")
    print(f"global_iou: " f"{global_iou:.6f}")
    print(f"global_dice: " f"{global_dice:.6f}")
    print(f"global_precision: " f"{global_precision:.6f}")
    print(f"global_recall: " f"{global_recall:.6f}")
    print(f"mean_image_iou: " f"{mean_image_iou:.6f} " f"+/- {std_image_iou:.6f}")
    print(f"mean_image_dice: " f"{mean_image_dice:.6f} " f"+/- {std_image_dice:.6f}")
    print(f"mean_image_precision: " f"{mean_image_precision:.6f} " f"+/- {std_image_precision:.6f}")
    print(f"mean_image_recall: " f"{mean_image_recall:.6f} " f"+/- {std_image_recall:.6f}")
    print(f"selection_score: " f"{selection_score:.6f}")
    print(f"best_selection_score: " f"{best_selection_score:.6f}")
    print(f"best_epoch: " f"{best_epoch}")
    
    # EARLY STOPPING
    
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping activated.")
        break


# FINAL SUMMARY

print("\n======== HEMOSET DEEPLABV3+ " "TRAINING COMPLETED ========")

print(f"Model: " f"{MODEL_NAME}")

print(f"Encoder: " f"{ENCODER_NAME}")

print(f"Segmentation mode: " f"{SEGMENTATION_MODE}")

print(f"Output channels: " f"{NUM_OUTPUT_CHANNELS}")

print(f"Best epoch: " f"{best_epoch}")

print(f"Best selection score: " f"{best_selection_score:.6f}")

print(f"Best global validation Dice: " f"{best_global_dice:.6f}")

print(f"Best mean-image validation Dice: " f"{best_mean_image_dice:.6f}")

print(f"Best checkpoint: " f"{checkpoint_path}")