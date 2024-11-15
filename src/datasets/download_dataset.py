from pathlib import Path
from typing import Optional

import pandas as pd
import wget

from src.consts import DATASETS_IDS, PATHS
import tempfile
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from src.types import DATASETS, SPLIT, DatasetArgs, TSplit
from datasets import load_dataset, DatasetDict

from datasets import load_dataset, DatasetDict

from src.consts import DATASETS_IDS, PATHS
from src.types import DATASETS
import random


def split_dataset(
    dataset: Dataset,
    num_splits: int,
    seed: int,
    split_ratio: Optional[float] = None,
    split_size: Optional[int] = None,
):
    """
    Split the dataset into multiple train splits and a test set.

    Args:
        dataset
        num_splits (int): Number of train splits.
        split_ratio (float): Proportion of data for each train split.
        split_size (int): Size of each train split.
        seed (int): Seed for shuffling to ensure reproducibility.

    Returns:
        DatasetDict: A dictionary containing the train splits and test set.
    """
    assert (split_ratio is None or split_size is None) and (
        split_ratio is not None or split_size is not None
    ), "Either split_ratio or split_size should be provided"

    # Shuffle the dataset
    dataset = dataset.shuffle(seed=seed)
    # Keep original indices while shuffling
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    dataset = dataset.select(indices)
    # Add original indices as a feature
    dataset = dataset.map(
        lambda example, idx: {"original_idx": indices[idx]}, with_indices=True
    )
    # Calculate sizes
    num_examples = len(dataset)
    if split_size is None:
        split_size = int(split_ratio * num_examples)

    # Create splits
    splits = {}
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size
        split_name = f"train{i+1}"
        splits[split_name] = dataset.select(range(start_idx, end_idx)).map(
            lambda x: {"split": split_name}
        )

    # Remaining data for test split
    remaining_start_idx = num_splits * split_size
    splits["test"] = dataset.select(range(remaining_start_idx, num_examples)).map(
        lambda x: {"split": "test"}
    )

    return DatasetDict(splits)


def load_knowns() -> Dataset:
    if not PATHS.RAW_KNOWN_1000_DIR.exists():
        with tempfile.TemporaryDirectory() as tmpdirname:
            wget.download(
                "https://rome.baulab.info/data/dsets/known_1000.json", out=tmpdirname
            )
            knowns_df = (
                pd.read_json(Path(tmpdirname) / "known_1000.json").set_index("known_id")
                # add space before values in the 'attribute' col to match counterfact
                .assign(**{"attribute": lambda x: " " + x["attribute"]})
            )

            dataset = dataset.rename_column("attribute", "target_true")
            dataset = Dataset.from_pandas(knowns_df)
            dataset.save_to_disk(str(PATHS.RAW_KNOWN_1000_DIR))

    return load_from_disk(str(PATHS.RAW_KNOWN_1000_DIR))  # type: ignore


def load_splitted_knowns(split: TSplit = (SPLIT.TRAIN1,)) -> Dataset:
    splitted_path = PATHS.PROCESSED_KNOWN_DIR / "splitted"

    if not splitted_path.exists():
        dataset = load_knowns()

        num_splits = 3
        split_ratio = 0.2
        seed = 42

        splitted_dataset = split_dataset(dataset, num_splits, split_ratio=split_ratio, seed=seed)
        splitted_dataset.save_to_disk(str(splitted_path))

    data = load_from_disk(str(splitted_path))  # type: ignore

    if split == "all":
        split = list(data.keys())
    if isinstance(split, str):
        split = [split]

    return concatenate_datasets([data[split] for split in split])


def load_knowns_pd() -> pd.DataFrame:
    return pd.DataFrame(load_knowns())


def load_splitted_counter_fact(split: TSplit = (SPLIT.TRAIN1,)) -> Dataset:
    splitted_path = PATHS.COUNTER_FACT_DIR / "splitted"

    if not splitted_path.exists():
        print("Creating splitted dataset")
        dataset_name = DATASETS_IDS[DATASETS.COUNTER_FACT]
        num_splits = 5
        split_size = 1500
        seed = 642

        dataset = load_dataset(dataset_name)["train"]

        splitted_dataset = split_dataset(
            dataset, num_splits, split_size=split_size, seed=seed
        )
        splitted_dataset.save_to_disk(str(splitted_path))

    data = load_from_disk(str(splitted_path))  # type: ignore

    if split == "all":
        split = list(data.keys())
    if isinstance(split, str):
        split = [split]

    return concatenate_datasets([data[split] for split in split])


def load_custom_dataset(dataset_args: DatasetArgs) -> Dataset:
    if dataset_args.name == DATASETS.KNOWN_1000:
        return load_splitted_knowns(dataset_args.splits)
    elif dataset_args.name == DATASETS.COUNTER_FACT:
        return load_splitted_counter_fact(dataset_args.splits)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_args.name}")
