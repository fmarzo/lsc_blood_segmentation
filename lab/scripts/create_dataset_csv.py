"""
file: create_dataset.py

brief:  this script it's used to create a .csv to label each frame of the dataset with a label (pig1, pig2, .. pig n)
        in this way we can choose whatever frames to put in the different set (train, validation or test). This will be launched
        ONE TIME to produce the entire .csv for all frames. Then, another script will read the file and create the 3 set.
"""
import glob
import os
import config_split
import csv

os.makedirs(config_split.OUT_DIR_SPLIT, exist_ok=True)

with open(config_split.CSV_FILE_NAME, 'w') as f:
    
    f.write(f"{config_split.IMG_STRING},{config_split.LABEL_STRING},{config_split.VIDEOID_STRING}\n")

    writer = csv.writer(f)
    for pig_number in range(1, config_split.NUM_DATASET_FOLDERS + 1):
        pig_name = f'pig{pig_number}'

        folder_dir_images = f'{config_split.DATASET_ROOT}/{pig_name}/imgs'
        folder_dir_labels = f'{config_split.DATASET_ROOT}/{pig_name}/labels'

        print(f'Reading {pig_name}')

        for image_path in sorted(glob.iglob(f'{folder_dir_images}/*')):

            if not image_path.endswith(config_split.IMG_EXT):
                continue

            image_filename = os.path.basename(image_path)
            name_without_extension = os.path.splitext(image_filename)[0]

            label_filename = f'{name_without_extension}_mask{config_split.IMG_EXT}'
            label_path = f'{folder_dir_labels}/{label_filename}'

            if not os.path.exists(label_path):
                print(f'Missing label for: {image_path}')
                print(f'Looking for: {label_path}')
                continue
                        
            writer.writerow([image_path, label_path, pig_name])

print(f'Created CSV file: {config_split.CSV_FILE_NAME}')