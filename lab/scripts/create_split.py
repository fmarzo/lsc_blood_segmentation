"""
file: create_split.py

brief:  this script reads the complete dataset.csv file and creates the three split files:
        train.csv, val.csv and test.csv.

        The split is performed by video_id, not by single frame. This means that all frames
        belonging to the same pig/video are assigned to the same set.

        This is important because frames from the same video are very similar to each other.
        If frames from the same video were split across train, validation and test, the model
        would be evaluated on images very close to those seen during training, causing data
        leakage and overly optimistic results.

        The output files are saved in the splits folder.
"""

import os
import pandas as pd
import config_split

os.makedirs(config_split.OUT_DIR_SPLIT, exist_ok=True)

df = pd.read_csv(config_split.CSV_FILE_NAME)

train_df = df[df["video_id"].isin(config_split.TRAIN_VIDEO_ID)]
val_df = df[df["video_id"].isin(config_split.VAL_VIDEO_ID)]
test_df = df[df["video_id"].isin(config_split.TEST_VIDEO_ID)]

train_df.to_csv(f"{config_split.OUT_DIR_SPLIT}/train.csv", index=False)
val_df.to_csv(f"{config_split.OUT_DIR_SPLIT}/val.csv", index=False)
test_df.to_csv(f"{config_split.OUT_DIR_SPLIT}/test.csv", index=False)

print("Train videos:", train_df["video_id"].unique())
print("Val videos:", val_df["video_id"].unique())
print("Test videos:", test_df["video_id"].unique())

print("Train samples:", len(train_df))
print("Val samples:", len(val_df))
print("Test samples:", len(test_df))