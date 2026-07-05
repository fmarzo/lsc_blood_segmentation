"""
file: hemoset_dataset.py

brief:  this class inherit torch Dataset interface to implement 3 methods to return 
        a pair of tensors ready for GPU-train.
        Accepts as parameters the file path and the tranformers object to use with the 
        file from the paths stored in the csv.
"""

import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform_img=None, transform_mask=None):
        self.csv_dirs = pd.read_csv(annotations_file)
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.csv_dirs)

    def __getitem__(self, row):
        img_path = self.csv_dirs.iloc[row, 0]
        image = Image.open(img_path)
        
        mask_path = self.csv_dirs.iloc[row, 1]
        mask = Image.open(mask_path)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
            # NB: loss will be Cross Entropy, so conversion to long data might be the case for this tensor  

        return image, mask