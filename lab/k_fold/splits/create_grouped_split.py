"""
Create stratified and grouped HemoSet split configurations.

Input:
    lab/splits/full_labeled_dataset.csv

Required columns:
    images
    labels
    video_id

For every selected configuration:
    - 6 PIG videos are used for training
    - 2 PIG videos are used for validation
    - 2 PIG videos are used for testing

Stratification:
    y = blood_bin, computed from the blood ratio of each mask.

Grouping:
    groups = video_id, so all frames from the same PIG always stay
    in the same split.

Output:
    lab/k_fold/generated_splits/config_000/train.csv
    lab/k_fold/generated_splits/config_000/validation.csv
    lab/k_fold/generated_splits/config_000/test.csv
    ...
"""

from pathlib import Path
import json
import random
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# PATHS
# ============================================================

LAB_DIRECTORY = Path(__file__).resolve().parents[2]

FULL_DATASET_CSV = (
    LAB_DIRECTORY
    / "splits"
    / "full_labeled_dataset.csv"
)

OUTPUT_DIRECTORY = (
    LAB_DIRECTORY
    / "k_fold"
    / "generated_splits"
)


# ============================================================
# CSV COLUMNS
# ============================================================

IMAGE_COLUMN = "images"
MASK_COLUMN = "labels"
VIDEO_COLUMN = "video_id"


# ============================================================
# SETTINGS
# ============================================================

NUMBER_OF_CONFIGURATIONS = 5

# Used to create candidate configurations.
NUMBER_OF_SPLIT_SEEDS = 500
BASE_SPLIT_SEED = 42

# Used to select an exactly balanced subset from the candidates.
NUMBER_OF_SELECTION_TRIALS = 20_000
SELECTION_SEED = 123

NUMBER_OF_BLOOD_BINS = 3

OVERWRITE_OUTPUT_DIRECTORY = True


if NUMBER_OF_CONFIGURATIONS % 5 != 0:
    raise ValueError(
        "NUMBER_OF_CONFIGURATIONS must be a multiple of 5. "
        "Examples: 5, 10, 15."
    )


# ============================================================
# DATASET
# ============================================================

def load_dataset():
    """Load and validate full_labeled_dataset.csv."""
    if not FULL_DATASET_CSV.is_file():
        raise FileNotFoundError(
            f"Dataset CSV not found: {FULL_DATASET_CSV}"
        )

    dataframe = pd.read_csv(
        FULL_DATASET_CSV
    )

    required_columns = {
        IMAGE_COLUMN,
        MASK_COLUMN,
        VIDEO_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Missing columns: {sorted(missing_columns)}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=[IMAGE_COLUMN]
        )
        .reset_index(drop=True)
    )

    dataframe[VIDEO_COLUMN] = (
        dataframe[VIDEO_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if dataframe[
        [
            IMAGE_COLUMN,
            MASK_COLUMN,
            VIDEO_COLUMN,
        ]
    ].isna().any().any():
        raise ValueError(
            "The CSV contains missing image, mask or video_id values."
        )

    return dataframe


def resolve_path(path_value):
    """Resolve an absolute or relative image/mask path."""
    path = Path(
        str(path_value)
    ).expanduser()

    candidates = [
        path,
        FULL_DATASET_CSV.parent / path,
        LAB_DIRECTORY / path,
        Path.cwd() / path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve path: {path_value}"
    )


# ============================================================
# BLOOD RATIO
# ============================================================

def compute_blood_ratio(mask_path):
    """Return foreground pixels divided by total mask pixels."""
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image)

    if mask.size == 0:
        raise ValueError(
            f"Empty mask: {mask_path}"
        )

    if mask.ndim == 2:
        blood_pixels = mask > 0
    else:
        blood_pixels = np.any(
            mask > 0,
            axis=-1,
        )

    return float(
        blood_pixels.mean()
    )


def add_blood_information(dataframe):
    """
    Add:
        blood_ratio
        blood_bin

    blood_bin is the stratification target y.
    """
    dataframe = dataframe.copy()

    blood_ratios = []

    for row_index, mask_value in enumerate(
        dataframe[MASK_COLUMN]
    ):
        mask_path = resolve_path(
            mask_value
        )

        blood_ratios.append(
            compute_blood_ratio(
                mask_path
            )
        )

        if (
            (row_index + 1) % 100 == 0
            or row_index + 1 == len(dataframe)
        ):
            print(
                f"Blood ratios: "
                f"{row_index + 1}/{len(dataframe)}",
                flush=True,
            )

    dataframe["blood_ratio"] = blood_ratios

    # qcut creates bins with approximately the same number of images.
    dataframe["blood_bin"] = pd.qcut(
        dataframe["blood_ratio"],
        q=NUMBER_OF_BLOOD_BINS,
        labels=False,
        duplicates="drop",
    )

    if dataframe["blood_bin"].isna().any():
        raise ValueError(
            "Some images could not be assigned to a blood bin."
        )

    dataframe["blood_bin"] = (
        dataframe["blood_bin"]
        .astype(int)
    )

    if dataframe["blood_bin"].nunique() < 2:
        raise ValueError(
            "Blood ratios contain too little variation "
            "for stratification."
        )

    print(
        "\nBlood-bin distribution:",
        flush=True,
    )

    print(
        dataframe["blood_bin"]
        .value_counts()
        .sort_index(),
        flush=True,
    )

    return dataframe


# ============================================================
# SPLIT HELPERS
# ============================================================

def get_pigs(
    dataframe,
    indices,
):
    """Return the sorted PIG identifiers represented by indices."""
    return tuple(
        sorted(
            dataframe.iloc[
                indices
            ][VIDEO_COLUMN]
            .unique()
            .tolist()
        )
    )


def get_bin_distribution(
    dataframe,
    indices,
):
    """Return the normalized blood-bin distribution."""
    all_bins = sorted(
        dataframe["blood_bin"]
        .unique()
        .tolist()
    )

    return (
        dataframe.iloc[
            indices
        ]["blood_bin"]
        .value_counts(
            normalize=True
        )
        .reindex(
            all_bins,
            fill_value=0.0,
        )
        .to_numpy(
            dtype=float
        )
    )


def compute_stratification_score(
    dataframe,
    train_indices,
    validation_indices,
    test_indices,
):
    """
    Compare the blood-bin distribution of every split with the
    complete-dataset distribution.

    Lower is better.
    """
    full_indices = (
        dataframe.index
        .to_numpy()
    )

    full_distribution = (
        get_bin_distribution(
            dataframe,
            full_indices,
        )
    )

    score = 0.0

    for split_indices in [
        train_indices,
        validation_indices,
        test_indices,
    ]:
        split_distribution = (
            get_bin_distribution(
                dataframe,
                split_indices,
            )
        )

        score += float(
            np.square(
                split_distribution
                - full_distribution
            ).sum()
        )

    return score


def configuration_key(configuration):
    """Return a unique key for one 6/2/2 configuration."""
    return (
        tuple(configuration["train_pigs"]),
        tuple(configuration["validation_pigs"]),
        tuple(configuration["test_pigs"]),
    )


def validate_configuration(configuration):
    """Validate group counts and absence of group leakage."""
    train_pigs = set(
        configuration["train_pigs"]
    )

    validation_pigs = set(
        configuration["validation_pigs"]
    )

    test_pigs = set(
        configuration["test_pigs"]
    )

    if len(train_pigs) != 6:
        return False

    if len(validation_pigs) != 2:
        return False

    if len(test_pigs) != 2:
        return False

    if not train_pigs.isdisjoint(
        validation_pigs
    ):
        return False

    if not train_pigs.isdisjoint(
        test_pigs
    ):
        return False

    if not validation_pigs.isdisjoint(
        test_pigs
    ):
        return False

    return True


# ============================================================
# CANDIDATE GENERATION WITH STRATIFIEDGROUPKFOLD
# ============================================================

def generate_candidates(dataframe):
    """
    Generate many valid 6/2/2 candidates.

    Outer split:
        8 PIGs for train + validation
        2 PIGs for test

    Inner split:
        6 PIGs for train
        2 PIGs for validation

    StratifiedGroupKFold may occasionally return a fold with a different
    number of groups because it optimizes class proportions. Those cases
    are simply ignored.
    """
    X = dataframe.index.to_numpy()
    y = dataframe["blood_bin"].to_numpy()
    groups = dataframe[VIDEO_COLUMN].to_numpy()

    candidates_by_key = {}

    for seed_offset in range(
        NUMBER_OF_SPLIT_SEEDS
    ):
        outer_seed = (
            BASE_SPLIT_SEED
            + seed_offset
        )

        outer_splitter = (
            StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=outer_seed,
            )
        )

        for outer_fold, (
            train_validation_indices,
            test_indices,
        ) in enumerate(
            outer_splitter.split(
                X,
                y,
                groups,
            )
        ):
            test_pigs = get_pigs(
                dataframe,
                test_indices,
            )

            if len(test_pigs) != 2:
                continue

            remaining = (
                dataframe
                .iloc[
                    train_validation_indices
                ]
                .reset_index()
                .rename(
                    columns={
                        "index": "_original_index"
                    }
                )
            )

            inner_X = (
                remaining.index
                .to_numpy()
            )

            inner_y = (
                remaining["blood_bin"]
                .to_numpy()
            )

            inner_groups = (
                remaining[VIDEO_COLUMN]
                .to_numpy()
            )

            inner_seed = (
                outer_seed * 100
                + outer_fold
            )

            inner_splitter = (
                StratifiedGroupKFold(
                    n_splits=4,
                    shuffle=True,
                    random_state=inner_seed,
                )
            )

            for (
                inner_train_indices,
                inner_validation_indices,
            ) in inner_splitter.split(
                inner_X,
                inner_y,
                inner_groups,
            ):
                train_indices = (
                    remaining
                    .iloc[
                        inner_train_indices
                    ]["_original_index"]
                    .to_numpy()
                )

                validation_indices = (
                    remaining
                    .iloc[
                        inner_validation_indices
                    ]["_original_index"]
                    .to_numpy()
                )

                configuration = {
                    "train_indices": train_indices,
                    "validation_indices": (
                        validation_indices
                    ),
                    "test_indices": test_indices,
                    "train_pigs": get_pigs(
                        dataframe,
                        train_indices,
                    ),
                    "validation_pigs": get_pigs(
                        dataframe,
                        validation_indices,
                    ),
                    "test_pigs": test_pigs,
                    "source_seed": outer_seed,
                    "outer_fold": outer_fold,
                }

                if not validate_configuration(
                    configuration
                ):
                    continue

                configuration[
                    "stratification_score"
                ] = compute_stratification_score(
                    dataframe,
                    train_indices,
                    validation_indices,
                    test_indices,
                )

                key = configuration_key(
                    configuration
                )

                previous = (
                    candidates_by_key.get(
                        key
                    )
                )

                if (
                    previous is None
                    or configuration[
                        "stratification_score"
                    ]
                    < previous[
                        "stratification_score"
                    ]
                ):
                    candidates_by_key[
                        key
                    ] = configuration

        if (
            (seed_offset + 1) % 50 == 0
            or seed_offset + 1
            == NUMBER_OF_SPLIT_SEEDS
        ):
            print(
                f"Split seeds: "
                f"{seed_offset + 1}/"
                f"{NUMBER_OF_SPLIT_SEEDS} | "
                f"unique candidates: "
                f"{len(candidates_by_key)}",
                flush=True,
            )

    candidates = list(
        candidates_by_key.values()
    )

    if len(candidates) < NUMBER_OF_CONFIGURATIONS:
        raise RuntimeError(
            f"Only {len(candidates)} valid candidates were found. "
            "Increase NUMBER_OF_SPLIT_SEEDS."
        )

    return candidates


# ============================================================
# EXACTLY BALANCED SELECTION
# ============================================================

def empty_role_counts(pig_ids):
    """Create zeroed train/validation/test counts for every PIG."""
    return {
        pig_id: {
            "train": 0,
            "validation": 0,
            "test": 0,
        }
        for pig_id in pig_ids
    }


def can_add_candidate(
    candidate,
    role_counts,
    target_train,
    target_validation,
    target_test,
):
    """Check whether adding a candidate exceeds any target count."""
    for pig_id in candidate[
        "train_pigs"
    ]:
        if (
            role_counts[pig_id]["train"]
            >= target_train
        ):
            return False

    for pig_id in candidate[
        "validation_pigs"
    ]:
        if (
            role_counts[pig_id]["validation"]
            >= target_validation
        ):
            return False

    for pig_id in candidate[
        "test_pigs"
    ]:
        if (
            role_counts[pig_id]["test"]
            >= target_test
        ):
            return False

    return True


def add_candidate_to_counts(
    candidate,
    role_counts,
):
    """Update role counts after choosing a candidate."""
    for pig_id in candidate[
        "train_pigs"
    ]:
        role_counts[pig_id]["train"] += 1

    for pig_id in candidate[
        "validation_pigs"
    ]:
        role_counts[pig_id][
            "validation"
        ] += 1

    for pig_id in candidate[
        "test_pigs"
    ]:
        role_counts[pig_id]["test"] += 1


def role_need_score(
    candidate,
    role_counts,
    target_train,
    target_validation,
    target_test,
):
    """
    Prefer candidates that assign PIGs to roles that still need them.

    Higher is better.
    """
    score = 0.0

    for pig_id in candidate[
        "train_pigs"
    ]:
        score += (
            target_train
            - role_counts[pig_id]["train"]
        )

    for pig_id in candidate[
        "validation_pigs"
    ]:
        score += 2.0 * (
            target_validation
            - role_counts[pig_id][
                "validation"
            ]
        )

    for pig_id in candidate[
        "test_pigs"
    ]:
        score += 2.0 * (
            target_test
            - role_counts[pig_id]["test"]
        )

    return score


def select_balanced_configurations(
    candidates,
    pig_ids,
):
    """
    Select configurations with exact role balance.

    With 10 configurations, each PIG must appear:
        - 6 times in training
        - 2 times in validation
        - 2 times in testing

    The search no longer requires five configurations to form one perfect
    round. This removes the overly strict condition that caused the previous
    RuntimeError.
    """
    target_train = (
        NUMBER_OF_CONFIGURATIONS
        * 6
        // len(pig_ids)
    )

    target_validation = (
        NUMBER_OF_CONFIGURATIONS
        * 2
        // len(pig_ids)
    )

    target_test = (
        NUMBER_OF_CONFIGURATIONS
        * 2
        // len(pig_ids)
    )

    random_generator = random.Random(
        SELECTION_SEED
    )

    best_selection = None
    best_score = float(
        "inf"
    )

    for trial in range(
        NUMBER_OF_SELECTION_TRIALS
    ):
        role_counts = empty_role_counts(
            pig_ids
        )

        selected = []
        selected_keys = set()

        used_validation_pairs = set()
        used_test_pairs = set()

        while (
            len(selected)
            < NUMBER_OF_CONFIGURATIONS
        ):
            feasible_candidates = []

            for candidate in candidates:
                key = configuration_key(
                    candidate
                )

                if key in selected_keys:
                    continue

                if not can_add_candidate(
                    candidate,
                    role_counts,
                    target_train,
                    target_validation,
                    target_test,
                ):
                    continue

                need_score = role_need_score(
                    candidate,
                    role_counts,
                    target_train,
                    target_validation,
                    target_test,
                )

                pair_penalty = 0.0

                validation_pair = tuple(
                    candidate[
                        "validation_pigs"
                    ]
                )

                test_pair = tuple(
                    candidate[
                        "test_pigs"
                    ]
                )

                if (
                    validation_pair
                    in used_validation_pairs
                ):
                    pair_penalty += 0.10

                if test_pair in used_test_pairs:
                    pair_penalty += 0.10

                # Lower ranking value is better.
                ranking_value = (
                    candidate[
                        "stratification_score"
                    ]
                    + pair_penalty
                    - 0.01 * need_score
                    + random_generator.random()
                    * 0.02
                )

                feasible_candidates.append(
                    (
                        ranking_value,
                        candidate,
                    )
                )

            if not feasible_candidates:
                break

            feasible_candidates.sort(
                key=lambda item: item[0]
            )

            # Randomly choose among the best few candidates to explore
            # different valid combinations across trials.
            shortlist_size = min(
                5,
                len(feasible_candidates),
            )

            _, chosen = (
                feasible_candidates[
                    random_generator.randrange(
                        shortlist_size
                    )
                ]
            )

            selected.append(
                chosen
            )

            selected_keys.add(
                configuration_key(
                    chosen
                )
            )

            used_validation_pairs.add(
                tuple(
                    chosen[
                        "validation_pigs"
                    ]
                )
            )

            used_test_pairs.add(
                tuple(
                    chosen[
                        "test_pigs"
                    ]
                )
            )

            add_candidate_to_counts(
                chosen,
                role_counts,
            )

        if (
            len(selected)
            != NUMBER_OF_CONFIGURATIONS
        ):
            continue

        exact_balance = all(
            counts["train"]
            == target_train
            and counts["validation"]
            == target_validation
            and counts["test"]
            == target_test
            for counts
            in role_counts.values()
        )

        if not exact_balance:
            continue

        selection_score = sum(
            candidate[
                "stratification_score"
            ]
            for candidate
            in selected
        )

        repeated_validation_pairs = (
            len(selected)
            - len(
                {
                    tuple(
                        candidate[
                            "validation_pigs"
                        ]
                    )
                    for candidate
                    in selected
                }
            )
        )

        repeated_test_pairs = (
            len(selected)
            - len(
                {
                    tuple(
                        candidate[
                            "test_pigs"
                        ]
                    )
                    for candidate
                    in selected
                }
            )
        )

        selection_score += (
            0.10
            * repeated_validation_pairs
        )

        selection_score += (
            0.10
            * repeated_test_pairs
        )

        if selection_score < best_score:
            best_score = (
                selection_score
            )

            best_selection = list(
                selected
            )

        if (
            (trial + 1) % 2000 == 0
            or trial + 1
            == NUMBER_OF_SELECTION_TRIALS
        ):
            print(
                f"Selection trials: "
                f"{trial + 1}/"
                f"{NUMBER_OF_SELECTION_TRIALS} | "
                f"best score: "
                f"{best_score if best_selection else 'not found'}",
                flush=True,
            )

    if best_selection is None:
        raise RuntimeError(
            "No exactly balanced subset was found. "
            "Increase NUMBER_OF_SPLIT_SEEDS and/or "
            "NUMBER_OF_SELECTION_TRIALS."
        )

    role_counts = empty_role_counts(
        pig_ids
    )

    for candidate in best_selection:
        add_candidate_to_counts(
            candidate,
            role_counts,
        )

    return (
        best_selection,
        role_counts,
    )


# ============================================================
# OUTPUT
# ============================================================

def prepare_output_directory():
    """Create a clean output directory."""
    if OUTPUT_DIRECTORY.exists():
        if not OVERWRITE_OUTPUT_DIRECTORY:
            raise FileExistsError(
                f"Output directory already exists: "
                f"{OUTPUT_DIRECTORY}"
            )

        shutil.rmtree(
            OUTPUT_DIRECTORY
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def bin_fractions(
    dataframe,
    indices,
):
    """Return the blood-bin fractions of one split."""
    fractions = (
        dataframe.iloc[
            indices
        ]["blood_bin"]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    return {
        f"bin_{int(bin_id)}_fraction": (
            float(fraction)
        )
        for bin_id, fraction
        in fractions.items()
    }


def save_outputs(
    dataframe,
    selected_configurations,
    role_counts,
    original_columns,
):
    """Save final split CSVs and summary files."""
    prepare_output_directory()

    dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "full_dataset_with_blood_information.csv",
        index=False,
    )

    summary_rows = []

    for configuration_index, configuration in enumerate(
        selected_configurations
    ):
        configuration_directory = (
            OUTPUT_DIRECTORY
            / f"config_{configuration_index:03d}"
        )

        configuration_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_dataframe = (
            dataframe
            .iloc[
                configuration[
                    "train_indices"
                ]
            ]
            .copy()
        )

        validation_dataframe = (
            dataframe
            .iloc[
                configuration[
                    "validation_indices"
                ]
            ]
            .copy()
        )

        test_dataframe = (
            dataframe
            .iloc[
                configuration[
                    "test_indices"
                ]
            ]
            .copy()
        )

        train_dataframe[
            original_columns
        ].to_csv(
            configuration_directory
            / "train.csv",
            index=False,
        )

        validation_dataframe[
            original_columns
        ].to_csv(
            configuration_directory
            / "validation.csv",
            index=False,
        )

        test_dataframe[
            original_columns
        ].to_csv(
            configuration_directory
            / "test.csv",
            index=False,
        )

        metadata = {
            "configuration_index": (
                configuration_index
            ),
            "source_seed": int(
                configuration[
                    "source_seed"
                ]
            ),
            "outer_fold": int(
                configuration[
                    "outer_fold"
                ]
            ),
            "stratification_score": float(
                configuration[
                    "stratification_score"
                ]
            ),
            "train_pigs": list(
                configuration[
                    "train_pigs"
                ]
            ),
            "validation_pigs": list(
                configuration[
                    "validation_pigs"
                ]
            ),
            "test_pigs": list(
                configuration[
                    "test_pigs"
                ]
            ),
            "train_images": int(
                len(train_dataframe)
            ),
            "validation_images": int(
                len(validation_dataframe)
            ),
            "test_images": int(
                len(test_dataframe)
            ),
            "train_mean_blood_ratio": float(
                train_dataframe[
                    "blood_ratio"
                ].mean()
            ),
            "validation_mean_blood_ratio": float(
                validation_dataframe[
                    "blood_ratio"
                ].mean()
            ),
            "test_mean_blood_ratio": float(
                test_dataframe[
                    "blood_ratio"
                ].mean()
            ),
            "train_bin_fractions": (
                bin_fractions(
                    dataframe,
                    configuration[
                        "train_indices"
                    ],
                )
            ),
            "validation_bin_fractions": (
                bin_fractions(
                    dataframe,
                    configuration[
                        "validation_indices"
                    ],
                )
            ),
            "test_bin_fractions": (
                bin_fractions(
                    dataframe,
                    configuration[
                        "test_indices"
                    ],
                )
            ),
        }

        with open(
            configuration_directory
            / "metadata.json",
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                metadata,
                metadata_file,
                indent=4,
            )

        summary_rows.append(
            {
                "configuration_index": (
                    configuration_index
                ),
                "source_seed": (
                    configuration[
                        "source_seed"
                    ]
                ),
                "outer_fold": (
                    configuration[
                        "outer_fold"
                    ]
                ),
                "stratification_score": (
                    configuration[
                        "stratification_score"
                    ]
                ),
                "train_pigs": ",".join(
                    configuration[
                        "train_pigs"
                    ]
                ),
                "validation_pigs": ",".join(
                    configuration[
                        "validation_pigs"
                    ]
                ),
                "test_pigs": ",".join(
                    configuration[
                        "test_pigs"
                    ]
                ),
                "train_images": int(
                    len(train_dataframe)
                ),
                "validation_images": int(
                    len(validation_dataframe)
                ),
                "test_images": int(
                    len(test_dataframe)
                ),
                "train_mean_blood_ratio": float(
                    train_dataframe[
                        "blood_ratio"
                    ].mean()
                ),
                "validation_mean_blood_ratio": float(
                    validation_dataframe[
                        "blood_ratio"
                    ].mean()
                ),
                "test_mean_blood_ratio": float(
                    test_dataframe[
                        "blood_ratio"
                    ].mean()
                ),
            }
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUTPUT_DIRECTORY
        / "selected_configurations.csv",
        index=False,
    )

    role_rows = []

    for pig_id, counts in sorted(
        role_counts.items()
    ):
        role_rows.append(
            {
                "video_id": pig_id,
                "train_count": (
                    counts["train"]
                ),
                "validation_count": (
                    counts["validation"]
                ),
                "test_count": (
                    counts["test"]
                ),
            }
        )

    pd.DataFrame(
        role_rows
    ).to_csv(
        OUTPUT_DIRECTORY
        / "pig_role_counts.csv",
        index=False,
    )


# ============================================================
# EXECUTION
# ============================================================

dataframe = load_dataset()

original_columns = list(
    dataframe.columns
)

dataframe = add_blood_information(
    dataframe
)

pig_ids = sorted(
    dataframe[VIDEO_COLUMN]
    .unique()
    .tolist()
)

if len(pig_ids) != 10:
    raise ValueError(
        "The script expects exactly 10 PIG groups. "
        f"Found {len(pig_ids)}: {pig_ids}"
    )


print(
    "\n======== DATASET ========",
    flush=True,
)

print(
    f"CSV: {FULL_DATASET_CSV}",
    flush=True,
)

print(
    f"Images: {len(dataframe)}",
    flush=True,
)

print(
    f"PIG groups: {pig_ids}",
    flush=True,
)

print(
    f"Requested configurations: "
    f"{NUMBER_OF_CONFIGURATIONS}",
    flush=True,
)


candidates = generate_candidates(
    dataframe
)


print(
    f"\nValid unique candidates: "
    f"{len(candidates)}",
    flush=True,
)


(
    selected_configurations,
    role_counts,
) = select_balanced_configurations(
    candidates,
    pig_ids,
)


save_outputs(
    dataframe,
    selected_configurations,
    role_counts,
    original_columns,
)


print(
    "\n======== SELECTED CONFIGURATIONS ========",
    flush=True,
)


for configuration_index, configuration in enumerate(
    selected_configurations
):
    print(
        f"\nConfiguration "
        f"{configuration_index:03d}",
        flush=True,
    )

    print(
        f"Train:      "
        f"{configuration['train_pigs']}",
        flush=True,
    )

    print(
        f"Validation: "
        f"{configuration['validation_pigs']}",
        flush=True,
    )

    print(
        f"Test:       "
        f"{configuration['test_pigs']}",
        flush=True,
    )

    print(
        f"Score:      "
        f"{configuration['stratification_score']:.6f}",
        flush=True,
    )


print(
    "\n======== PIG ROLE COUNTS ========",
    flush=True,
)


for pig_id, counts in sorted(
    role_counts.items()
):
    print(
        f"{pig_id}: "
        f"train={counts['train']}, "
        f"validation={counts['validation']}, "
        f"test={counts['test']}",
        flush=True,
    )


print(
    f"\nOutput directory: "
    f"{OUTPUT_DIRECTORY}",
    flush=True,
)