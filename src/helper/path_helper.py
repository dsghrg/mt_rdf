import os
from enum import Enum
from pathlib import Path

from src.query.query_type import QueryType

DATA_QUERY_WDBENCH_PATH = os.path.join('data', 'queries', 'wdbench')
MODEL_PATH = os.path.join('models')
DATASET_RAW_PATH = os.path.join('data', 'raw')
DATASET_PROCESSED_PATH = os.path.join('data', 'processed')



def file_exists_or_create(file_path: str) -> bool:
    file_path = Path(file_path)
    file_exists = os.path.isfile(file_path)

    if not file_exists:
        file_path.parent.mkdir(exist_ok=True, parents=True)

    return file_exists

# /data/processed
def dataset_processed_folder_path(dataset_name: str, seed: int = None) -> str:
    if seed:
        return os.path.join(DATASET_PROCESSED_PATH, dataset_name, f"seed_{seed}")
    return os.path.join(DATASET_PROCESSED_PATH, dataset_name)

def dataset_processed_file_path(dataset_name: str, file_name: str, seed: int = None) -> str:
    return os.path.join(dataset_processed_folder_path(dataset_name, seed=seed), file_name)

# /data/raw
def dataset_raw_file_path(file_name: str) -> str:
    return os.path.join(DATASET_RAW_PATH, file_name)

def get_file_name_from_qtype(qtype: QueryType):
    return f'{qtype.value}.txt'


def wdbench_query_path(qtype: QueryType):
    return os.path.join(DATA_QUERY_WDBENCH_PATH, get_file_name_from_qtype(qtype=qtype))

# /models
def experiment_file_path(experiment_name: str, file_name: str) -> str:
    return os.path.join(MODEL_PATH, experiment_name, file_name)

def experiment_config_path(experiment_name: str) -> str:
    return experiment_file_path(experiment_name, 'config.cfg')
