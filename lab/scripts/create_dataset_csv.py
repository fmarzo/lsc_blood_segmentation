"""
file: create_dataset.py

brief:  this script it's used to create a .csv to label each frame of the dataset with a label (pig1, pig2, .. pig n)
        in this way we can choose whatever frames to put in the different set (train, validation or test). This will be launched
        ONE TIME to produce the entire .csv for all frames. Then, another script will read the file and create the 3 set.
"""
import glob
import os
from src.config_split import * 
import csv

counter_value = 0
os.makedirs(OUT_DIR_SPLIT, exist_ok=True)

with open(CSV_FILE_PATH, 'w') as f:
    writer = csv.writer(f)
    
    f.write(f"{IMG_STRING},{LABEL_STRING},{VIDEOID_STRING}\n")

    for pig_number in range(1, NUM_DATASET_FOLDERS + 1):
        pig_name = f'pig{pig_number}'

        folder_dir_images = f'{DATASET_ROOT}/{pig_name}/imgs'
        folder_dir_labels = f'{DATASET_ROOT}/{pig_name}/labels'

        print(f'Reading {pig_name}')

        for image_path in sorted(glob.iglob(f'{folder_dir_images}/*')):

            if not image_path.endswith(IMG_EXT):
                continue

            image_filename = os.path.basename(image_path)
            name_without_extension = os.path.splitext(image_filename)[0]

            label_filename = f'{name_without_extension}_mask{IMG_EXT}'
            label_path = f'{folder_dir_labels}/{label_filename}'

            if not os.path.exists(label_path):
                print(f'Missing label for: {image_path}')
                print(f'Looking for: {label_path}')
                continue
                        
            writer.writerow([image_path, label_path, pig_name])
            counter_value += 1
print(f'Created CSV file: {CSV_FILE_PATH}')
print(f'Rows: {counter_value}')