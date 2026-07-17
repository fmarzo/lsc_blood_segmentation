"""
file: evaluate_bleed_seg.py

brief:
    This script loads the best validation checkpoint of a trained U-Net or
    U-Net++ model and evaluates it on the Bleeding Segmentation v1.0 test set
    without data augmentation or gradient computation.

    The model checkpoints used for this evaluation were trained on HemoSet.
    Therefore, this script performs a zero-shot cross-dataset evaluation.

    For each test image, the script stores the true positives, false positives,
    false negatives and true negatives separately. It then computes the IoU
    and Dice score of the blood class for every image.

    Dataset-level metrics are computed by summing the confusion statistics
    across all test images before calculating IoU and Dice.

    Per-image metrics are also computed separately and reported as mean and
    standard deviation. Both methods consider only the blood class and exclude
    the background score.

    ZERO SHOT OF A MODEL TRAINED ON HEMOSET WITH A TEST SET OF RABBANI
"""

import torch
import segmentation_models_pytorch as smp
import os

from torch.utils.data import DataLoader
from src import config_split
from src.data_transforms import create_bleed_eval_transform
from src.hemoset_dataset_v2 import CustomImageDataset


"""
function: create_model

brief:
    This routine creates the selected model and returns the path
    of its best validation checkpoint.
"""
def create_model(model):
    if model == "unet_plus_plus":
        unet_plus = smp.UnetPlusPlus(
            encoder_name=config_split.ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=config_split.NUM_CLASSES,
        ).to("cuda")

        checkpoint_path = os.path.join(
            config_split.MODEL_PRETRAINED_DIR,
            f"unet_plus_plus_{config_split.SEGMENTATION_MODE}_best_{config_split.ENCODER_NAME}.pth",
        )

        return unet_plus, checkpoint_path

    else:
        unet = smp.Unet(
            encoder_name=config_split.ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=config_split.NUM_CLASSES,
        ).to("cuda")

        checkpoint_path = os.path.join(
            config_split.MODEL_PRETRAINED_DIR,
            f"unet_{config_split.SEGMENTATION_MODE}_best_{config_split.ENCODER_NAME}.pth",
        )

        return unet, checkpoint_path


"""
function: prepare_mask

brief:
    This routine prepares the mask for the evaluation.
"""
def prepare_mask(test_mask, mode):
    if mode == "binary":
        return test_mask.float().to("cuda")
    else:
        return torch.squeeze(
            test_mask,
            1,
        ).to(torch.long).to("cuda")


"""
function: get_predictions

brief:
    This routine converts the model output into the final predicted map.
"""
def get_predictions(logits, mode):
    if mode == "binary":
        return (
            torch.sigmoid(logits) >= config_split.BINARY_THRESHOLD
        ).long()
    else:
        return logits.argmax(dim=1)


"""
function: get_segmentation_stats

brief:
    This routine computes TP, FP, FN and TN for every image.
"""
def get_segmentation_stats(predictions, mask, mode):
    if mode == "binary":
        return smp.metrics.get_stats(
            predictions,
            mask.long(),
            mode=mode,
        )
    else:
        return smp.metrics.get_stats(
            predictions,
            mask,
            mode=mode,
            num_classes=config_split.NUM_CLASSES,
        )


"""
function: get_class_target

brief:
    This routine returns the index representing the blood class.
"""
def get_class_target(mode):
    if mode == "binary":
        return 0
    else:
        return 1


# The installed cuDNN version does not support the Tesla K80 GPU.
torch.backends.cudnn.enabled = False

# Disable NNPACK to avoid unsupported hardware warnings on the CPU node.
torch.backends.nnpack.set_flags(False)


model_name = config_split.MODEL_TO_EVALUATE


# Instantiate the model and get the checkpoint path.
model, checkpoint_path = create_model(model_name)


# Check that the checkpoint exists.
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
    )


# Load the weights saved at the best HemoSet validation epoch.
model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location="cuda",
    )
)


# Set the model to evaluation mode.
model.eval()


# Prepare data for iteration.
eval_transform = create_bleed_eval_transform()


# Load the Bleeding Segmentation v1.0 test split.
test_ds = CustomImageDataset(
    config_split.CSV_TEST_PATH_V1P0,
    eval_transform,
)


# Check that the test split contains at least one image.
if len(test_ds) == 0:
    raise ValueError(
        f"Empty test dataset: {config_split.CSV_TEST_PATH_V1P0}"
    )


test_bleed_DL = DataLoader(
    test_ds,
    batch_size=4,
    num_workers=2,
    shuffle=False,
)


tp_batches = []
fp_batches = []
fn_batches = []
tn_batches = []


# Disable gradient computation because evaluation does not update the model.
with torch.no_grad():
    for test_img, test_mask in test_bleed_DL:

        test_mask = prepare_mask(
            test_mask,
            config_split.SEGMENTATION_MODE,
        )

        test_img = test_img.to("cuda")

        img_forward = model(test_img)

        final_map = get_predictions(
            img_forward,
            config_split.SEGMENTATION_MODE,
        )

        batch_tp, batch_fp, batch_fn, batch_tn = (
            get_segmentation_stats(
                final_map,
                test_mask,
                mode=config_split.SEGMENTATION_MODE,
            )
        )

        # Store TP, FP, FN and TN for each image in the batch.
        tp_batches.append(batch_tp.cpu())
        fp_batches.append(batch_fp.cpu())
        fn_batches.append(batch_fn.cpu())
        tn_batches.append(batch_tn.cpu())


# Join the statistics from all batches while keeping the results
# of each test image separate.
tp = torch.cat(tp_batches, dim=0)
fp = torch.cat(fp_batches, dim=0)
fn = torch.cat(fn_batches, dim=0)
tn = torch.cat(tn_batches, dim=0)


# With reduction="none", obtain one metric for each image and class.
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


# Select the blood class according to the segmentation mode.
blood_class_index = get_class_target(
    config_split.SEGMENTATION_MODE
)


# Sum the confusion statistics across all test images while keeping
# the classes separate.
global_tp = tp.sum(dim=0)
global_fp = fp.sum(dim=0)
global_fn = fn.sum(dim=0)
global_tn = tn.sum(dim=0)


# Compute one dataset-level score for each class.
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


# Select the global scores corresponding to the blood class.
global_iou = global_iou_classes[
    blood_class_index
].item()

global_dice = global_dice_classes[
    blood_class_index
].item()


# Keep one blood segmentation score for each test image.
iou_per_image = iou_classes[:, blood_class_index]
dice_per_image = dice_classes[:, blood_class_index]


# Count the ground-truth and predicted blood pixels for each test image.
blood_gt_pixels = (
    tp[:, blood_class_index] +
    fn[:, blood_class_index]
)

blood_pred_pixels = (
    tp[:, blood_class_index] +
    fp[:, blood_class_index]
)


# Separate images containing blood from images with an empty blood mask.
images_with_blood_mask = blood_gt_pixels > 0
empty_images_mask = blood_gt_pixels == 0


total_images = iou_per_image.numel()
images_with_blood = images_with_blood_mask.sum().item()
empty_images = empty_images_mask.sum().item()


# Check whether the model correctly predicts no blood on empty images.
correct_empty_predictions = (
    empty_images_mask &
    (blood_pred_pixels == 0)
).sum().item()


# Count empty images where the model incorrectly predicts blood.
empty_images_with_false_positives = (
    empty_images_mask &
    (blood_pred_pixels > 0)
).sum().item()


# Compute the mean per-image scores across the test set.
mean_iou = iou_per_image.mean().item()
mean_dice = dice_per_image.mean().item()


# Measure how much the per-image scores vary across the test set.
std_iou = iou_per_image.std(correction=0).item()
std_dice = dice_per_image.std(correction=0).item()


print("\n----- Evaluation configuration -----")
print(f"Model: {model_name}")
print(f"Checkpoint: {checkpoint_path}")
print(f"Test CSV: {config_split.CSV_TEST_PATH_V1P0}")

print("\n----- Test image composition -----")
print(f"Total images: {total_images}")
print(f"Images with blood: {images_with_blood}")
print(f"Images without blood: {empty_images}")


print("\n----- Empty image predictions -----")
print(f"Correctly predicted as empty: {correct_empty_predictions}")
print(
    "Empty images with false-positive blood: "
    f"{empty_images_with_false_positives}"
)


print("\n----- Dataset-level metrics -----")
print(f"IoU:  {global_iou:.4f}")
print(f"Dice: {global_dice:.4f}")


print("\n----- Per-image metrics -----")
print(f"IoU:  {mean_iou:.4f} +/- {std_iou:.4f}")
print(f"Dice: {mean_dice:.4f} +/- {std_dice:.4f}")
