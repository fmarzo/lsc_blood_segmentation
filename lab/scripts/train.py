"""
file: train.py

brief:  this script acts as "main" entry for python calls

        Creates trasnformers to convert both image and mask from pgn to torch.tensor
        Use of mean and std to normalize data comes off knowns ResNet pre trained values
        
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from src.hemoset_dataset import CustomImageDataset
from src import config_split
from torch.utils.data import DataLoader

"""
function: tensor_proprieties
brief:    this routine displays all the useful proprieties of a torch tensor (shape, data type.. etc)
"""
def tensor_proprieties (tensor, index):
    print(f"shape: {tensor[index].shape}")
    print(f"type: {tensor[index].dtype}")
    print(f"shape {tensor[index].shape}")
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

train_features, train_labels = next(iter(train_hemo_DL))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")



