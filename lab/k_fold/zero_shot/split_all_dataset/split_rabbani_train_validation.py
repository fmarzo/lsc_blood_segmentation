"""
file: split_rabbani_train_validation.py

brief:  split the complete Rabbani CSV into stratified train and validation
        CSV files for full-dataset zero-shot training.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

random_seed = 42
validation_fraction = 0.20
number_of_blood_bins = 3
lab_directory = Path(__file__).resolve().parents[3]
full_dataset_csv = lab_directory / "splits_v1p0" / "full_labeled_dataset.csv"
output_directory = (
    lab_directory / "k_fold" / "zero_shot" / "split_dataset" / "generated" / "rabbani"
)


def compute_blood_ratio(mask_path):
    with Image.open(mask_path) as image:
        mask = np.asarray(image)
    blood = np.any(mask > 0, axis=2) if mask.ndim == 3 else mask > 0
    return float(blood.mean())


if not full_dataset_csv.is_file():
    raise FileNotFoundError(f"Rabbani CSV not found: {full_dataset_csv}")

df = pd.read_csv(full_dataset_csv)
required_columns = ["images", "labels", "video_id"]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

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

splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=validation_fraction,
    random_state=random_seed,
)
train_indices, validation_indices = next(splitter.split(df, df["blood_bin"]))
train_df = df.iloc[train_indices].copy()
validation_df = df.iloc[validation_indices].copy()
if set(train_df["images"]) & set(validation_df["images"]):
    raise RuntimeError("Train and validation images overlap.")

output_directory.mkdir(parents=True, exist_ok=True)
train_df[original_columns].to_csv(output_directory / "train.csv", index=False)
validation_df[original_columns].to_csv(output_directory / "validation.csv", index=False)
metadata = {
    "dataset": "rabbani",
    "random_seed": random_seed,
    "validation_fraction": validation_fraction,
    "train_images": len(train_df),
    "validation_images": len(validation_df),
    "train_mean_blood_ratio": float(train_df["blood_ratio"].mean()),
    "validation_mean_blood_ratio": float(validation_df["blood_ratio"].mean()),
    "train_blood_bins": {int(k): int(v) for k, v in train_df["blood_bin"].value_counts().sort_index().to_dict().items()},
    "validation_blood_bins": {int(k): int(v) for k, v in validation_df["blood_bin"].value_counts().sort_index().to_dict().items()},
}
with open(output_directory / "metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)
print(f"train images: {len(train_df)}")
print(f"validation images: {len(validation_df)}")
print(f"train blood bins: {metadata['train_blood_bins']}")
print(f"validation blood bins: {metadata['validation_blood_bins']}")
print(f"output directory: {output_directory}")
