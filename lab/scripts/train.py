"""
file: train.py

brief:  this script acts as "main" entry for python calls

        Creates trasnformers to convert both image and mask from pgn to torch.tensor
        Use of mean and std to normalize data comes off knowns ResNet pre trained values
        
"""

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.hemoset_dataset import CustomImageDataset
from src import config_split
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

"""
function: tensor_proprieties
brief:    this routine displays all the useful proprieties of a torch tensor (shape, data type.. etc)
"""
def tensor_proprieties (tensor, index):
    print(f"shape: {tensor[index].shape}")
    print(f"type: {tensor[index].dtype}")
    print(f"values: {torch.unique(tensor[index])}")
    print(f"len: {len(tensor[index])}")


# creating the transform Compose for images and masks
transform_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )])

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

unet = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=2)
unet.to("cuda")

# images 
train_img = train_img.to("cuda")
img_1 = unet(train_img)
print(img_1.shape)

# mask
# uses index to avoid removal batch size of 1
train_mask = torch.squeeze(train_mask,1).to(torch.long).to("cuda")

# loss
ce_loss = torch.nn.CrossEntropyLoss().to("cuda")
print(ce_loss(img_1, train_mask))

# optimizer Adam

adam = torch.optim.Adam(unet.parameters(), lr=0.001)

#testing overfit on the same batch (4): does the model learn?
# for i in range(300):
#     adam.zero_grad()
#     img_1 = unet(train_img)
#     loss = ce_loss(img_1, train_mask)
#     if i % 50 == 0:
#         print(loss)
#     loss.backward()
#     adam.step()

# testing all dataset for few epochs
n_epochs = 3
for epoch in range (n_epochs):
    i = 0
    for train_img, train_mask in train_hemo_DL:
        adam.zero_grad()
        train_mask = torch.squeeze(train_mask,1).to(torch.long).to("cuda")
        train_img = train_img.to("cuda")
        img_forward = unet(train_img)
        loss = ce_loss(img_forward, train_mask)
        if i % 50 == 0:
            print(loss)
        loss.backward()
        adam.step()
        i += 1