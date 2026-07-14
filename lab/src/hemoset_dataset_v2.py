"""
file: hemoset_dataset.py

brief:  this class inherit torch Dataset interface to implement 3 methods to return 
        a pair of tensors ready for GPU-train.
        Accepts as parameters the file path and the tranformers object to use with the 
        file from the paths stored in the csv.
"""

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform = None):
        self.csv_dirs = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.csv_dirs)

    def __getitem__(self, row):
        img_path = self.csv_dirs.iloc[row, 0]
        image = Image.open(img_path).convert("RGB")
        
        mask_path = self.csv_dirs.iloc[row, 1]
        mask = Image.open(mask_path)

        image = F.to_image(image)
        mask = tv_tensors.Mask(F.pil_to_tensor(mask))

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask # NB: loss will be Cross Entropy, so conversion to long data might be the case for this tensor  
