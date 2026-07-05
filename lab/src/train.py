import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import hemoset_dataset
#from  scripts.config_split import * 

transform_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )])

transform_mask = transforms.Compose([transforms.PILToTensor()])

CSV_FILE_NAME = "/homes/fmarzo/cvcs2026/lsc_blood_segmentation/lab/splits/full_labeled_dataset.csv"

hemo_ds = hemoset_dataset.CustomImageDataset(CSV_FILE_NAME, transform_img, transform_mask)

# testing methods of the dataset class
print(len(hemo_ds))
print (hemo_ds[0])


