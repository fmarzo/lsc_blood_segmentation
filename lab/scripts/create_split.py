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
from src.config_split import *

def check_data_integrity ():
        train_set = set(TRAIN_VIDEO_ID)
        val_set   = set(VAL_VIDEO_ID)
        test_set  = set(TEST_VIDEO_ID)

        #check for same tag in different sets
        if not train_set.isdisjoint(val_set):
                print(f"Overlap train/val: {train_set & val_set}")
                return False
        if not train_set.isdisjoint(test_set):
                print(f"Overlap train/test: {train_set & test_set}")
                return False
        if not val_set.isdisjoint(test_set):
                print(f"Overlap val/test: {val_set & test_set}")
                return False
        
        #check for duplicates
        for name, lst in [("train", TRAIN_VIDEO_ID), ("val", VAL_VIDEO_ID),("test", TEST_VIDEO_ID)]:
                if len(lst) != len(set(lst)):
                        print(f"Duplicate found! {name}: {lst}")
                        return False

        set_len = len(train_set) + len (val_set) + len(test_set)
        if set_len != NUM_DATASET_FOLDERS:
                print (f"WARNING! found {set_len} dataset folder: are you sure about dataset split config?")
        return True

os.makedirs(OUT_DIR_SPLIT, exist_ok=True)
df = pd.read_csv(CSV_FILE_PATH)

#check if for some reason dataset it's repeated in two different sets
if check_data_integrity():
        train_df = df[df[VIDEOID_STRING].isin(TRAIN_VIDEO_ID)]
        val_df = df[df[VIDEOID_STRING].isin(VAL_VIDEO_ID)]
        test_df = df[df[VIDEOID_STRING].isin(TEST_VIDEO_ID)]

        train_df.to_csv(f"{OUT_DIR_SPLIT}/train.csv", index=False)
        val_df.to_csv(f"{OUT_DIR_SPLIT}/val.csv", index=False)
        test_df.to_csv(f"{OUT_DIR_SPLIT}/test.csv", index=False)

        print ("======== FINAL RESULTS ===========")
        print("Train videos:", train_df[VIDEOID_STRING].unique())
        print("Val videos:", val_df[VIDEOID_STRING].unique())
        print("Test videos:", test_df[VIDEOID_STRING].unique())

        print("Train samples:", len(train_df))
        print("Val samples:", len(val_df))
        print("Test samples:", len(test_df))

