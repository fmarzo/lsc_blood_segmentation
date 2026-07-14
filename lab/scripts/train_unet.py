"""
file: train_unet.py

brief:  this script is the main entry point for training the blood segmentation model.

    The model uses a ResNet-18 encoder pretrained on ImageNet. During training, 
    both the pretrained encoder and the U-Net decoder parameters are updated.

    The image transform converts each input image from PIL format to a PyTorch
    tensor and normalizes its RGB channels using the ImageNet mean and standard
    deviation expected by the pretrained ResNet encoder.

    The mask transform converts each segmentation mask to a tensor without
    normalizing it, preserving its class values.

    CustomImageDataset reads the image and mask paths from the corresponding
    CSV split and returns matching image-mask pairs after applying the selected
    transforms.

    DataLoader groups these pairs into batches and loads them during training,
    validation, and testing. Training samples are shuffled at every epoch,
    while validation and test samples keep their original order.

    The script supports two segmentation approaches:

    - Multiclass segmentation:
      the model produces two output channels, one for background and one for
      blood. CrossEntropyLoss compares the predicted class of every pixel with
      the corresponding ground-truth class.

    - Binary segmentation:
      the model produces one output channel representing the presence of blood.
      BCEWithLogitsLoss evaluates every pixel independently, while DiceLoss
      checks how well the entire predicted blood region overlaps the real one.

      DiceLoss is especially useful in this dataset because blood pixels can be
      much fewer than background pixels. It prevents the model from obtaining a
      good result simply by predicting mostly background and encourages it to
      correctly recover the shape and area of the blood region.
"""

import os
import sys
import torch
import torchvision.transforms as transforms
from src.hemoset_dataset import CustomImageDataset
from src import config_split
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# # COMMANDS USED ONLY TO RUN THE TRAINING FROM THE BASH SCRIPT
# # The installed cuDNN version does not support the Tesla K80 GPU.
# torch.backends.cudnn.enabled = False

# # Disable NNPACK to avoid unsupported hardware warnings on the CPU node.
# torch.backends.nnpack.set_flags(False)

"""
function: prepare mask
brief:    this routine prepares the mask for the loss
"""  
def prepare_mask(train_mask, mode):
    if mode == "binary":
        return train_mask.float().to('cuda')
    else:
        return torch.squeeze(train_mask,1).to(torch.long).to("cuda") # uses index to avoid removal batch size of 1


"""
function: compute_loss
brief:    this routine computes the selected loss depending on mode
"""    
def compute_loss(mode, logits, mask, bce_loss, dice_loss, ce_loss):
    if mode == "binary":
        bce_value = bce_loss(logits, mask)
        dice_value = dice_loss(logits, mask)
        return bce_value + dice_value
    else:    
        return ce_loss(logits, mask)


"""
function: get_predictions
brief:    this routine converts the model output into the final predicted map of value
"""
def get_predictions(logits, mode):
    if mode == "binary":
        return (torch.sigmoid(logits)>=config_split.BINARY_THRESHOLD).long()
    else:
        return logits.argmax(dim = 1)


"""
function: get_segmentation_stats
brief:    this routine retrieves the metrics for segmentation
"""
def get_segmentation_stats(predictions, mask, mode):
    if mode == "binary":
        return smp.metrics.get_stats(predictions, mask.long(), mode=mode)
    else:
        return smp.metrics.get_stats(predictions, mask, mode=mode, num_classes=config_split.NUM_CLASSES)
    
"""
function: get_class_target
brief:    this routine retrieves the mask for only target class (blood)
"""
def get_class_target (target_class, mode):
    if mode == "binary":
        return target_class[0]
    else:
        return target_class[1]

"""
function: tensor_proprieties
brief:    this routine displays all the useful proprieties of a torch tensor (shape, data type.. etc)
"""
def tensor_proprieties(tensor, index):
    print(f"shape: {tensor[index].shape}")
    print(f"type: {tensor[index].dtype}")
    print(f"values: {torch.unique(tensor[index])}")
    print(f"len: {len(tensor[index])}")

# creating the transform Compose for images and masks
transform_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )])

# check for number of epochs, if not passed, default is 5
if len (sys.argv) > 1:
    n_epochs = int(sys.argv[1])
else:
    n_epochs = config_split.DEFAULT_EPOCHS

transform_mask = transforms.Compose([transforms.PILToTensor()])

train_ds = CustomImageDataset(config_split.CSV_TRAIN_PATH, transform_img, transform_mask)
valid_ds = CustomImageDataset(config_split.CSV_VALID_PATH, transform_img, transform_mask)
test_ds  = CustomImageDataset(config_split.CSV_TEST_PATH,  transform_img, transform_mask)

train_hemo_DL = DataLoader(train_ds, 4, num_workers=2, shuffle=True)
valid_hemo_DL = DataLoader(valid_ds, 4, num_workers=2, shuffle=False)
test_hemo_DL  = DataLoader(test_ds,  4, num_workers=2, shuffle=False)

train_img, train_mask = next(iter(train_hemo_DL))
print(f"Feature batch shape: {train_img.size()}")
print(f"Labels batch shape: {train_mask.size()}")

# instantiate the unet
unet = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=config_split.NUM_CLASSES)
unet.to("cuda")

# TEST FOR IMAGE SHAPE
# # images 
# train_img = train_img.to("cuda")
# img_1 = unet(train_img)
# print(img_1.shape)

# INITIAL TEST ON A SINGLE BATCH
# # mask
# train_mask = prepare_mask(train_mask, config_split.SEGMENTATION_MODE)

# # select loss_train
# loss_value = compute_loss(config_split.SEGMENTATION_MODE, img_1, train_mask)
# print(loss_value)

# optimizer Adam
adam = torch.optim.Adam(unet.parameters(), lr=0.001)

os.makedirs(
    os.path.dirname(config_split.UNET_PRETRAINED_PATH),
    exist_ok=True
)

best_val_loss = float('inf')

# Initialize the loss functions before the training loop so they are instantiated only once
bce_loss = torch.nn.BCEWithLogitsLoss().to("cuda")
dice_loss = smp.losses.DiceLoss(mode=config_split.SEGMENTATION_MODE, from_logits=True).to("cuda")
ce_loss = torch.nn.CrossEntropyLoss().to("cuda")

# testing all dataset for few epochs
for epoch in range (n_epochs):
    print(f"------------ EPOCH: {epoch+1} ------------")
    i = 0
    train_loss_sum = 0
    for train_img, train_mask in train_hemo_DL:
        adam.zero_grad()
        train_mask = prepare_mask(train_mask, config_split.SEGMENTATION_MODE)
        train_img = train_img.to("cuda")
        img_forward = unet(train_img)
        loss_train_value = compute_loss(config_split.SEGMENTATION_MODE, img_forward, train_mask, bce_loss, dice_loss, ce_loss)
        if i % 50 == 0:
            print(f"loss_train {loss_train_value}")
        train_loss_sum += loss_train_value.item()
        loss_train_value.backward()
        adam.step()
        i += 1
    train_batch_num = len(train_hemo_DL)
    avg_train_loss = train_loss_sum/train_batch_num

    unet.eval()
    with torch.no_grad():
        j = 0
        val_loss_sum = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        for val_img, val_mask in valid_hemo_DL:
            val_mask = prepare_mask(val_mask, config_split.SEGMENTATION_MODE)
            val_img = val_img.to("cuda")
            img_forward = unet(val_img)
            loss_valid_value = compute_loss(config_split.SEGMENTATION_MODE, img_forward, val_mask, bce_loss, dice_loss, ce_loss)
            if j % 50 == 0:
                print(f"loss_valid {loss_valid_value}")
            val_loss_sum += loss_valid_value.item()
            final_map = get_predictions(img_forward, config_split.SEGMENTATION_MODE)
            batch_tp, batch_fp, batch_fn, batch_tn = get_segmentation_stats(final_map, val_mask, mode=config_split.SEGMENTATION_MODE)
            tp += batch_tp.sum(dim=0)
            fp += batch_fp.sum(dim=0)
            fn += batch_fn.sum(dim=0)
            tn += batch_tn.sum(dim=0)
            j += 1

    val_batch_num = len(valid_hemo_DL)
    avg_val_loss = val_loss_sum/val_batch_num
    avg_iou_classes = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
    avg_dice_classes = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")

    # retrieving scores only for BLOOD, excluding background pixels
    avg_iou = get_class_target(avg_iou_classes, mode=config_split.SEGMENTATION_MODE)
    avg_dice = get_class_target(avg_dice_classes, mode=config_split.SEGMENTATION_MODE)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(unet.state_dict(), config_split.UNET_PRETRAINED_PATH)

    unet.train()

    print(f"----- Avg values for epoch {epoch+1} -----")

    print(f"avg_train_loss {avg_train_loss}")
    print(f"avg_val_loss {avg_val_loss}")
    print(f"avg_iou {avg_iou}")
    print(f"avg_dice {avg_dice}")
    