from typing import Any, Iterable, Literal
from typing import Dict
from typing import NamedTuple
from typing import NewType
from typing import Optional
from typing import TypedDict, Union

from dataclasses import dataclass

from src.utils.types_utils import STREnum


class SPLIT(STREnum):
    TRAIN1 = "train1"
    TRAIN2 = "train2"
    TRAIN3 = "train3"
    TRAIN4 = "train4"
    TRAIN5 = "train5"
    TEST = "test"


class MODEL_ARCH(STREnum):
    LLAMA2 = "llama2"
    LLAMA3_1 = "llama3_1"
    LLAMA3_2 = "llama3_2"
    MISTRAL = "mistral"
    PHI = "phi"


class DATASETS(STREnum):
    KNOWN_1000 = "known_1000"
    COUNTER_FACT = "counter_fact"


TModelID = NewType("TModelID", str)
TDatasetID = NewType("TDatasetID", str)
TSplit = Union[SPLIT, Iterable[SPLIT], Literal["all"]]


@dataclass(frozen=True)  # Make the dataclass immutable
class DatasetArgs:
    name: DATASETS
    splits: TSplit = "all"

    def __post_init__(self):
        if self.splits != "all" and isinstance(self.splits, str):
            # Since the class is frozen, we need to use object.__setattr__
            object.__setattr__(self, "splits", [self.splits])

    @property
    def dataset_name(self) -> str:
        split_name = ""
        if self.splits != "all":
            split_name = f"_{self.splits}"
        return self.name + split_name

    def copy_with_splits(self, splits: TSplit) -> "DatasetArgs":
        """Create a new DatasetArgs instance with different splits but same dataset name."""
        return DatasetArgs(name=self.name, splits=splits)

    def __hash__(self):
        # Convert splits to tuple if it's a list for hashing
        splits = tuple(self.splits) if isinstance(self.splits, list) else self.splits
        return hash((self.name, splits))

    def __eq__(self, other):
        if not isinstance(other, DatasetArgs):
            return NotImplemented
        return self.name == other.name and (
            self.splits == other.splits
            if isinstance(self.splits, str)
            else list(self.splits) == list(other.splits)
        )
