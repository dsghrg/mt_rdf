import os
from enum import Enum
from pathlib import Path

from src.query.query_type import QueryType

DATA_QUERY_WDBENCH_PATH = os.path.join('data', 'queries', 'wdbench')


def get_file_name_from_qtype(qtype: QueryType):
    return f'{qtype.value}.txt'


def wdbench_query_path(qtype: QueryType):
    return os.path.join(DATA_QUERY_WDBENCH_PATH, get_file_name_from_qtype(qtype=qtype))
