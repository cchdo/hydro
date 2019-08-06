from dataclasses import dataclass, asdict, astuple, field
from functools import singledispatch
from typing import Union
from datetime import date, time

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

fmat = logging.Formatter(
    "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(fmat)

log.addHandler(ch)

WOCECTD_to_ARGO = {1: 0, 2: 1, 3: 3, 4: 4, 5: 9, 6: 0, 7: 0, 9: 9}
WOCECTD_to_ODV = {1: 0, 2: 0, 3: 4, 4: 8, 5: 1, 6: 1, 7: 1, 9: 1}


def _union_checker(union, obj):
    log.debug(f"Checking typing.Union types")
    union_types = union.__args__
    log.debug(f"found types {union_types}")
    for dtype in union_types:
        log.debug(f"checking if {obj} is {dtype}")
        if isinstance(obj, dtype):
            return True
    log.debug(f"{type(obj)} not in {union}")
    return False


class ValidateInitTypes:
    def __post_init__(self):
        for attr, dtype in self.__annotations__.items():
            log.debug(f"Checking {attr} is {dtype}")
            obj = getattr(self, attr)
            self_type = type(obj)
            error = (
                f"Invalid data type for {self!r}: The type of {attr} "
                f"is expected to be {dtype} not {self_type}"
            )
            if "typing.Union" in str(dtype):
                log.debug(f"typing.Union found, checking allowed types")
                if not _union_checker(dtype, obj):
                    raise TypeError(error)
            elif self_type is not dtype:
                raise TypeError(error)
        log.debug(f"{self} looks ok")


class ToAndFromDict:
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass(frozen=True)
class ExchangeCompositeKey(ValidateInitTypes, ToAndFromDict):
    station: str
    cast: int
    sample: str  # may be the pressure value for CTD data


@dataclass(frozen=True)
class ExchangeTimestamp(ValidateInitTypes, ToAndFromDict):
    date: date
    time: Union[time, type(None)]