"""
file: create_dataset_v1p0.py

brief:
    This script is used to create a CSV file containing the path of each image
    and its corresponding segmentation mask.

    The dataset contains two folders:

        images/
        masks/

    The script is executed one time to create the complete CSV file.
    Another script can later split it into train, validation and test sets.
"""

import glob
import os
import csv

from src.config_split import *


counter_value = 0

# Create the output directory if it does not exist
if not os.path.exists(OUT_DIR_SPLIT_V1P0):
    os.makedirs(OUT_DIR_SPLIT_V1P0)

# Binary mode avoids empty lines in CSV files with Python 2
with open(CSV_FILE_PATH_V1P0, 'wb') as f:
    writer = csv.writer(f)

    writer.writerow([
        IMG_STRING,
        LABEL_STRING,
        VIDEOID_STRING
    ])

    folder_dir_images = os.path.join(
        BLEEDING_DATASET_ROOT,
        'images'
    )

    folder_dir_masks = os.path.join(
        BLEEDING_DATASET_ROOT,
        'masks'
    )

    print('Reading dataset: {}'.format(BLEEDING_DATASET_ROOT))

    image_search_path = os.path.join(folder_dir_images, '*')

    for image_path in sorted(glob.iglob(image_search_path)):

        if not image_path.lower().endswith(BLEEDING_IMG_EXT):
            continue

        image_filename = os.path.basename(image_path)
        name_without_extension = os.path.splitext(image_filename)[0]

        mask_filename = '{}{}'.format(
            name_without_extension,
            BLEEDING_MASK_EXT
        )

        mask_path = os.path.join(
            folder_dir_masks,
            mask_filename
        )

        if not os.path.exists(mask_path):
            print('Missing mask for: {}'.format(image_path))
            print('Looking for: {}'.format(mask_path))
            continue

        writer.writerow([
            image_path,
            mask_path,
            BLEEDING_VIDEO_ID
        ])

        counter_value += 1

print('Created CSV file: {}'.format(CSV_FILE_PATH_V1P0))
print('Rows: {}'.format(counter_value))