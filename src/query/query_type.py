from enum import Enum


class QueryType(str, Enum):
    single_bgps = 'single_bgps'
    multiple_bgps = 'multiple_bgps'
    paths = 'paths'
    c2rpqs = 'c2rpqs'
    opts = 'opts'

    @classmethod
    def get(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No member of {cls.__name__} has value '{value}'")