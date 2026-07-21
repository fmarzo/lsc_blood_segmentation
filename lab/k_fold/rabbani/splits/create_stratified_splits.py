"""
file: create_stratified_splits.py

brief:  this script creates repeated stratified train, validation and test
        configurations for the Rabbani bleeding segmentation dataset.

    The complete dataset is read from:

        lab/splits_v1p0/full_labeled_dataset.csv

    The dataset does not contain patient, PIG, clip or useful video groups.
    Every row has the same video_id, so grouped splitting cannot be applied.

    For this reason, every configuration is created with two consecutive
    StratifiedShuffleSplit operations at image level:

    1. the complete dataset is divided into 650 train-validation images and
       the remaining images are assigned to the test set;

    2. the 650 train-validation images are divided into 525 training images
       and 125 validation images.

    Stratification is based on blood_bin. blood_bin is obtained from the blood
    ratio of every segmentation mask.

    Masks without blood are kept in a separate bin only when there are enough
    empty masks to distribute that bin across train, validation and test.
    Otherwise, the empty masks are merged into the lowest blood-ratio quantile.

    In the current Rabbani dataset there is only one empty mask, so three
    balanced blood-ratio quantiles are used.

    Different random seeds generate different configurations while preserving
    the same train, validation and test sizes and similar blood-ratio
    distributions.

    Positional argument:

        first argument: number of configurations to create

    When no argument is provided, the script creates 5 configurations.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


# ============================================================
# GENERAL SETTINGS
# ============================================================

train_size = 525
validation_size = 125
train_validation_size = train_size + validation_size

default_number_of_configurations = 5
base_random_seed = 42

number_of_positive_blood_bins = 3

# The script is located in:
# lab/k_fold/rabbani/splits/create_stratified_splits.py
lab_directory = Path(__file__).resolve().parents[3]

full_dataset_csv = (
    lab_directory
    / "splits_v1p0"
    / "full_labeled_dataset.csv"
)

output_directory = (
    lab_directory
    / "k_fold"
    / "rabbani"
    / "generated_splits"
)


# ============================================================
# INPUT ARGUMENT
# ============================================================

if len(sys.argv) > 1 and sys.argv[1] != "":
    number_of_configurations = int(sys.argv[1])
else:
    number_of_configurations = default_number_of_configurations

if len(sys.argv) > 2:
    raise ValueError(
        "Usage: python -m k_fold.rabbani.splits.create_stratified_splits "
        "[number_of_configurations]"
    )

if number_of_configurations <= 0:
    raise ValueError("The number of configurations must be greater than zero.")


"""
function: compute_blood_ratio
brief:    this routine computes the percentage of blood pixels in one mask
"""
def compute_blood_ratio(mask_path):
    mask = np.asarray(Image.open(mask_path))

    if mask.ndim == 3:
        blood_pixels = np.any(mask > 0, axis=2)
    else:
        blood_pixels = mask > 0

    return float(blood_pixels.mean())


"""
function: create_blood_bins
brief:    this routine creates categorical blood-ratio bins for stratification
"""
def create_blood_bins(dataframe):
    blood_ratio = dataframe["blood_ratio"]

    zero_mask = blood_ratio == 0
    zero_count = int(zero_mask.sum())

    test_size = len(dataframe) - train_validation_size
    smallest_split_size = min(train_size, validation_size, test_size)

    # Minimum approximate class size required to represent one bin in every set.
    minimum_separate_bin_count = int(
        np.ceil(len(dataframe) / smallest_split_size)
    )

    # Keep empty masks in a separate bin only when that bin is large enough.
    if zero_count >= minimum_separate_bin_count:
        blood_bins = pd.Series(
            np.zeros(len(dataframe), dtype=int),
            index=dataframe.index,
        )

        positive_mask = blood_ratio > 0
        positive_ratios = blood_ratio.loc[positive_mask]

        number_of_unique_ratios = positive_ratios.nunique()

        if number_of_unique_ratios == 0:
            return blood_bins

        number_of_bins = min(
            number_of_positive_blood_bins,
            number_of_unique_ratios,
        )

        positive_bins = pd.qcut(
            positive_ratios,
            q=number_of_bins,
            labels=False,
            duplicates="drop",
        )

        blood_bins.loc[positive_mask] = positive_bins.astype(int) + 1

        print(
            f"empty masks kept in a separate bin: "
            f"{zero_count} images"
        )

        return blood_bins

    # A singleton or very small empty-mask bin cannot be stratified safely.
    # Merge it into the lowest blood-ratio quantile.
    number_of_unique_ratios = blood_ratio.nunique()

    number_of_bins = min(
        number_of_positive_blood_bins,
        number_of_unique_ratios,
    )

    blood_bins = pd.qcut(
        blood_ratio,
        q=number_of_bins,
        labels=False,
        duplicates="drop",
    )

    print(
        f"empty masks merged into the lowest blood-ratio bin: "
        f"{zero_count} images"
    )

    return blood_bins.astype(int)


"""
function: print_split_information
brief:    this routine prints split size and blood distribution information
"""
def print_split_information(split_name, split_dataframe):
    print(
        f"{split_name}: "
        f"images={len(split_dataframe)}, "
        f"mean_blood_ratio={split_dataframe['blood_ratio'].mean():.6f}"
    )

    print(
        f"{split_name} blood bins: "
        f"{split_dataframe['blood_bin'].value_counts().sort_index().to_dict()}"
    )


# ============================================================
# CREATE STRATIFIED CONFIGURATIONS
# ============================================================

if not full_dataset_csv.is_file():
    raise FileNotFoundError(f"Dataset CSV not found: {full_dataset_csv}")

full_dataframe = pd.read_csv(full_dataset_csv)

required_columns = ["images", "labels", "video_id"]

missing_columns = [
    column
    for column in required_columns
    if column not in full_dataframe.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in {full_dataset_csv}: {missing_columns}"
    )

if len(full_dataframe) <= train_validation_size:
    raise ValueError(
        f"The dataset contains {len(full_dataframe)} images, "
        f"but at least {train_validation_size + 1} are required."
    )

original_columns = full_dataframe.columns.tolist()

print(f"dataset: {full_dataset_csv}")
print(f"total images: {len(full_dataframe)}")
print("computing blood ratio for every mask")

full_dataframe["blood_ratio"] = full_dataframe["labels"].apply(
    compute_blood_ratio
)

full_dataframe["blood_bin"] = create_blood_bins(full_dataframe)

blood_bin_counts = (
    full_dataframe["blood_bin"]
    .value_counts()
    .sort_index()
)

test_size = len(full_dataframe) - train_validation_size
smallest_split_size = min(train_size, validation_size, test_size)

minimum_bin_count = int(
    np.ceil(len(full_dataframe) / smallest_split_size)
)

if (blood_bin_counts < minimum_bin_count).any():
    raise ValueError(
        "Every blood bin must be large enough to be represented in train, "
        "validation and test. "
        f"Minimum count: {minimum_bin_count}. "
        f"Current counts: {blood_bin_counts.to_dict()}"
    )

if output_directory.exists():
    shutil.rmtree(output_directory)

output_directory.mkdir(parents=True, exist_ok=True)

full_information_path = (
    output_directory
    / "full_dataset_with_blood_information.csv"
)

full_dataframe.to_csv(full_information_path, index=False)

configuration_summaries = []

for configuration_id in range(number_of_configurations):
    random_seed = base_random_seed + configuration_id

    outer_split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_validation_size,
        test_size=test_size,
        random_state=random_seed,
    )

    train_validation_indices, test_indices = next(
        outer_split.split(
            full_dataframe,
            full_dataframe["blood_bin"],
        )
    )

    train_validation_dataframe = full_dataframe.iloc[
        train_validation_indices
    ].copy()

    test_dataframe = full_dataframe.iloc[test_indices].copy()

    inner_split = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        test_size=validation_size,
        random_state=random_seed + 10000,
    )

    train_indices, validation_indices = next(
        inner_split.split(
            train_validation_dataframe,
            train_validation_dataframe["blood_bin"],
        )
    )

    train_dataframe = train_validation_dataframe.iloc[
        train_indices
    ].copy()

    validation_dataframe = train_validation_dataframe.iloc[
        validation_indices
    ].copy()

    train_paths = set(train_dataframe["images"])
    validation_paths = set(validation_dataframe["images"])
    test_paths = set(test_dataframe["images"])

    if train_paths & validation_paths:
        raise RuntimeError("Train and validation sets overlap.")

    if train_paths & test_paths:
        raise RuntimeError("Train and test sets overlap.")

    if validation_paths & test_paths:
        raise RuntimeError("Validation and test sets overlap.")

    if len(train_dataframe) != train_size:
        raise RuntimeError("The generated training size is not correct.")

    if len(validation_dataframe) != validation_size:
        raise RuntimeError("The generated validation size is not correct.")

    if len(test_dataframe) != test_size:
        raise RuntimeError("The generated test size is not correct.")

    configuration_directory = (
        output_directory
        / f"config_{configuration_id:03d}"
    )

    configuration_directory.mkdir(parents=True, exist_ok=True)

    train_dataframe[original_columns].to_csv(
        configuration_directory / "train.csv",
        index=False,
    )

    validation_dataframe[original_columns].to_csv(
        configuration_directory / "validation.csv",
        index=False,
    )

    test_dataframe[original_columns].to_csv(
        configuration_directory / "test.csv",
        index=False,
    )

    metadata = {
        "configuration_id": configuration_id,
        "random_seed": random_seed,
        "train_images": len(train_dataframe),
        "validation_images": len(validation_dataframe),
        "test_images": len(test_dataframe),
        "train_mean_blood_ratio": train_dataframe["blood_ratio"].mean(),
        "validation_mean_blood_ratio": validation_dataframe["blood_ratio"].mean(),
        "test_mean_blood_ratio": test_dataframe["blood_ratio"].mean(),
        "train_blood_bins": (
            train_dataframe["blood_bin"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "validation_blood_bins": (
            validation_dataframe["blood_bin"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "test_blood_bins": (
            test_dataframe["blood_bin"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }

    with open(
        configuration_directory / "metadata.json",
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    configuration_summaries.append(metadata)

    print("")
    print("============================================================")
    print(f"CONFIGURATION: {configuration_id:03d}")
    print("============================================================")

    print_split_information("train", train_dataframe)
    print_split_information("validation", validation_dataframe)
    print_split_information("test", test_dataframe)

summary_dataframe = pd.DataFrame(configuration_summaries)

summary_dataframe.to_csv(
    output_directory / "split_summary.csv",
    index=False,
)

print("")
print(f"generated configurations: {number_of_configurations}")
print(f"output directory: {output_directory}")