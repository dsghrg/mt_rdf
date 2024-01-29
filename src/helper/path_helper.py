import os
from enum import Enum
from pathlib import Path


DATA_QUERY_WDBENCH_PATH = os.path.join('data', 'queries', 'wdbench')


class QueryType(str, Enum):
    single_bgps = 'single_bgps'
    multiple_bgps = 'multiple_bgps'
    paths = 'paths'
    c2rpqs = 'c2rpqs'
    opts = 'opts'


def get_file_name_from_qtype(qtype: QueryType):
    return f'{qtype.value}.txt'


def wdbench_query_path(qtype: QueryType):
    return os.path.join(DATA_QUERY_WDBENCH_PATH, get_file_name_from_qtype(qtype=qtype))
