"""
file: split_hemoset_train_validation.py

brief:  split the complete HemoSet CSV into grouped, stratified train and
        validation CSV files for full-dataset zero-shot training.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

random_seed = 42
number_of_folds = 5
selected_fold = 0
number_of_blood_bins = 3
lab_directory = Path(__file__).resolve().parents[3]
full_dataset_csv = lab_directory / "splits" / "full_labeled_dataset.csv"
output_directory = (
    lab_directory / "k_fold" / "zero_shot" / "split_dataset" / "generated" / "hemoset"
)


def compute_blood_ratio(mask_path):
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    blood = np.any(mask > 0, axis=2) if mask.ndim == 3 else mask > 0
    return float(blood.mean())


if not full_dataset_csv.is_file():
    raise FileNotFoundError(f"HemoSet CSV not found: {full_dataset_csv}")

df = pd.read_csv(full_dataset_csv)
required_columns = ["images", "labels", "video_id"]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
if df["video_id"].nunique() < number_of_folds:
    raise ValueError(f"At least {number_of_folds} video groups are required.")

original_columns = df.columns.tolist()
print(f"dataset: {full_dataset_csv}")
print(f"total images: {len(df)}")
print("computing blood ratio for every mask")
df["blood_ratio"] = df["labels"].apply(compute_blood_ratio)
df["blood_bin"] = pd.qcut(
    df["blood_ratio"],
    q=min(number_of_blood_bins, df["blood_ratio"].nunique()),
    labels=False,
    duplicates="drop",
).astype(int)

splitter = StratifiedGroupKFold(
    n_splits=number_of_folds,
    shuffle=True,
    random_state=random_seed,
)
train_indices, validation_indices = list(
    splitter.split(df, df["blood_bin"], groups=df["video_id"])
)[selected_fold]
train_df = df.iloc[train_indices].copy()
validation_df = df.iloc[validation_indices].copy()
train_groups = set(train_df["video_id"])
validation_groups = set(validation_df["video_id"])
if train_groups & validation_groups:
    raise RuntimeError("Train and validation groups overlap.")

output_directory.mkdir(parents=True, exist_ok=True)
train_df[original_columns].to_csv(output_directory / "train.csv", index=False)
validation_df[original_columns].to_csv(output_directory / "validation.csv", index=False)
metadata = {
    "dataset": "hemoset",
    "random_seed": random_seed,
    "number_of_folds": number_of_folds,
    "selected_fold": selected_fold,
    "train_images": len(train_df),
    "validation_images": len(validation_df),
    "train_groups": sorted(str(value) for value in train_groups),
    "validation_groups": sorted(str(value) for value in validation_groups),
    "train_mean_blood_ratio": float(train_df["blood_ratio"].mean()),
    "validation_mean_blood_ratio": float(validation_df["blood_ratio"].mean()),
}
with open(output_directory / "metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)
print(f"train images: {len(train_df)}")
print(f"validation images: {len(validation_df)}")
print(f"train groups: {sorted(train_groups)}")
print(f"validation groups: {sorted(validation_groups)}")
print(f"output directory: {output_directory}")
