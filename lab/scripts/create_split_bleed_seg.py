"""
file: create_split_v1p0.py

brief:
    This script reads the complete bleeding segmentation dataset CSV
    and creates train.csv, val.csv and test.csv.

    The split is randomly performed by image using a fixed random seed
    to make the result reproducible.
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from src.config_split import *


# Create the output directory if it does not exist
if not os.path.exists(OUT_DIR_SPLIT_V1P0):
    os.makedirs(OUT_DIR_SPLIT_V1P0)


df = pd.read_csv(CSV_FILE_PATH_V1P0)

# Check that the dataset is not empty
if len(df) == 0:
    raise ValueError(
        'The dataset CSV is empty: {}'.format(CSV_FILE_PATH_V1P0)
    )


# 70% train, 15% validation and 15% test
train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=42,
    shuffle=True
)


train_df.to_csv(CSV_TRAIN_PATH_V1P0, index=False)
val_df.to_csv(CSV_VALID_PATH_V1P0, index=False)
test_df.to_csv(CSV_TEST_PATH_V1P0, index=False)

print('Total samples: {}'.format(len(df)))
print('Train samples: {}'.format(len(train_df)))
print('Validation samples: {}'.format(len(val_df)))
print('Test samples: {}'.format(len(test_df)))

print('Train CSV: {}'.format(CSV_TRAIN_PATH_V1P0))
print('Validation CSV: {}'.format(CSV_VALID_PATH_V1P0))
print('Test CSV: {}'.format(CSV_TEST_PATH_V1P0))