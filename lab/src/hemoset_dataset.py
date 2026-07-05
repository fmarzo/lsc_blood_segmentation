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

    def __getitem__(self, idx-):
        img_path = self.csv_dirs.iloc[idx, 0]
        image = Image.open(img_path)
        
        mask_path = self.csv_dirs.iloc[idx, 1]
        mask = Image.open(mask_path)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask