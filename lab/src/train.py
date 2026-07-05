import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import hemoset_dataset
#from  scripts.config_split import 

# Data Loader 

img_path = f'/work/cvcs2026/latent_space_cowboys/datasets/HemoSet/pig10/imgs/000180.png'
mask_path = f'/work/cvcs2026/latent_space_cowboys/datasets/HemoSet/pig10/labels/000180_mask.png'

#IMAGE STANDARDIZATION 

print ("========== IMG STD =============")

img = Image.open(img_path)
arr_img = np.array(img)
print(arr_img.shape)

transform_img = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )])

img_tensor = transform_img(img)

print(img_tensor.shape)
print(img_tensor)

#MASK STANDARDIZATION 

print ("========== MASK STD =============")
mask = Image.open(mask_path)
arr_mask = np.array(mask)
print(arr_mask.shape)

transform_mask = transforms.Compose([transforms.PILToTensor()])
mask_tensor = transform_mask(mask)

print(mask_tensor.shape)
print(mask_tensor.unique())
print(mask_tensor.dtype)

# loss will be Cross Entropy, so convert to long data type the mask tensor
mask_tensor = mask_tensor.to(dtype=torch.long)

CSV_PATH = "/homes/fmarzo/cvcs2026/lsc_blood_segmentation/lab/splits/full_labeled_dataset.csv"

hemo_ds = hemoset_dataset.CustomImageDataset(CSV_PATH, transform_img, transform_mask)

print(len(hemo_ds))

print (hemo_ds[0])


