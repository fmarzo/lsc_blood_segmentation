"""
file: create_split_v1p0.py

brief:
    This script reads the complete Rabbani bleeding segmentation CSV and
    creates deterministic train, validation and test splits while preserving
    the original row order.

    The CSV header is not counted as a data row.

    Split assignment:

    - train:      data rows 1 to 525
    - validation: data rows 526 to 650
    - test:       data rows 651 to the end

    No random shuffling is performed.
"""

import os

import pandas as pd

from src.config_split import (
    CSV_FILE_PATH_V1P0,
    CSV_TRAIN_PATH_V1P0,
    CSV_VALID_PATH_V1P0,
    CSV_TEST_PATH_V1P0,
    OUT_DIR_SPLIT_V1P0,
)


# ============================================================
# SPLIT BOUNDARIES
# ============================================================

# Human-readable CSV data rows:
#
# train:      1-525
# validation: 526-650
# test:       651-end
#
# Pandas uses zero-based indexing and an exclusive upper boundary.
TRAIN_END_INDEX = 525
VALIDATION_END_INDEX = 650


# ============================================================
# LOAD COMPLETE DATASET
# ============================================================

os.makedirs(
    OUT_DIR_SPLIT_V1P0,
    exist_ok=True,
)


df = pd.read_csv(
    CSV_FILE_PATH_V1P0,
)


if len(df) == 0:
    raise ValueError(
        f"The dataset CSV is empty: {CSV_FILE_PATH_V1P0}"
    )


if len(df) <= VALIDATION_END_INDEX:
    raise ValueError(
        "The complete dataset must contain more than "
        f"{VALIDATION_END_INDEX} samples, but only "
        f"{len(df)} were found."
    )


# ============================================================
# ORDERED SPLIT
# ============================================================

# Data rows 1-525.
train_df = df.iloc[
    :TRAIN_END_INDEX
].copy()


# Data rows 526-650.
val_df = df.iloc[
    TRAIN_END_INDEX:VALIDATION_END_INDEX
].copy()


# Data rows 651-end.
test_df = df.iloc[
    VALIDATION_END_INDEX:
].copy()


# Reset the row indices inside each output CSV.
train_df.reset_index(
    drop=True,
    inplace=True,
)

val_df.reset_index(
    drop=True,
    inplace=True,
)

test_df.reset_index(
    drop=True,
    inplace=True,
)


# ============================================================
# CONSISTENCY CHECKS
# ============================================================

if len(train_df) != 525:
    raise RuntimeError(
        f"Expected 525 training samples, found {len(train_df)}."
    )


if len(val_df) != 125:
    raise RuntimeError(
        f"Expected 125 validation samples, found {len(val_df)}."
    )


if (
    len(train_df)
    + len(val_df)
    + len(test_df)
    != len(df)
):
    raise RuntimeError(
        "The split sample counts do not match the complete dataset."
    )


# ============================================================
# SAVE SPLITS
# ============================================================

train_df.to_csv(
    CSV_TRAIN_PATH_V1P0,
    index=False,
)

val_df.to_csv(
    CSV_VALID_PATH_V1P0,
    index=False,
)

test_df.to_csv(
    CSV_TEST_PATH_V1P0,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "======== RABBANI ORDERED SPLIT ========"
)

print(
    f"Total samples: {len(df)}"
)

print(
    f"Train samples: {len(train_df)} "
    "(complete CSV data rows 1-525)"
)

print(
    f"Validation samples: {len(val_df)} "
    "(complete CSV data rows 526-650)"
)

print(
    f"Test samples: {len(test_df)} "
    f"(complete CSV data rows 651-{len(df)})"
)

print(
    f"Train CSV: {CSV_TRAIN_PATH_V1P0}"
)

print(
    f"Validation CSV: {CSV_VALID_PATH_V1P0}"
)

print(
    f"Test CSV: {CSV_TEST_PATH_V1P0}"
)