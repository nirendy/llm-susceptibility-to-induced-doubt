import os
from pathlib import Path
from typing import NamedTuple

from src.types import DATASETS, MODEL_ARCH, TModelID, TDatasetID


class PATHS:
    PROJECT_DIR = Path(__file__).parent.parent.resolve()
    DATA_DIR = PROJECT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"
    RAW_KNOWN_1000_DIR = RAW_DATA_DIR / DATASETS.KNOWN_1000
    COUNTER_FACT_DIR = PREPROCESSED_DATA_DIR / DATASETS.COUNTER_FACT
    DATA_SHARED_DIR = PROJECT_DIR / "shared"
    RUNS_DIR = PROJECT_DIR / "runs"
    TENSORBOARD_DIR = PROJECT_DIR / "tensorboard"
    OUTPUT_DIR = PROJECT_DIR / "output"
    SLURM_DIR = PROJECT_DIR / "slurm"


class ENV_VARS:
    MASTER_PORT = "MASTER_PORT"
    MASTER_ADDR = "MASTER_ADDR"


class FORMATS:
    TIME = "%Y%m%d_%H-%M-%S"
    LOGGER_FORMAT = "%(asctime)s - %(message)s"


class DDP:
    MASTER_PORT = os.environ.get(ENV_VARS.MASTER_PORT, "12355")
    MASTER_ADDR = "localhost"
    BACKEND = "nccl"
    SHUFFLE = True
    DROP_LAST = True
    NUM_WORKERS = 0


MODEL_SIZES_PER_ARCH_TO_MODEL_ID: dict[MODEL_ARCH, dict[str, TModelID]] = {
    MODEL_ARCH.LLAMA2: {
        "8B": "meta-llama/Llama-3.1-8B",
    },
    MODEL_ARCH.LLAMA3_1: {
        "8B": "meta-llama/Llama-3.1-8B",
    },
    MODEL_ARCH.LLAMA3_2: {
        "1B": "meta-llama/Llama-3.2-1B",
        "3B": "meta-llama/Llama-3.2-3B",
    },
    MODEL_ARCH.MISTRAL: {
        "8x7B": "mistralai/Mixtral-8x7B-v0.1",
        "Nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    },
    MODEL_ARCH.PHI: {
        "3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    },
}  # type: ignore

MODEL_IDS_TO_ARCH_AND_SIZE: dict[TModelID, tuple[MODEL_ARCH, str]] = {
    model_id: (arch, size)
    for arch, sizes in MODEL_SIZES_PER_ARCH_TO_MODEL_ID.items()
    for size, model_id in sizes.items()
}


DATASETS_IDS: dict[DATASETS, TDatasetID] = {
    DATASETS.COUNTER_FACT: "NeelNanda/counterfact-tracing"
}  # type: ignore
