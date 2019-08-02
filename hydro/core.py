from dataclasses import dataclass, asdict, astuple, field
from enum import IntEnum
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


class ExchangeFlagCTD(IntEnum):
    NOFLAG = 0  # no idea if this will cause issue
    UNCALIBRATED = 1
    GOOD = 2
    QUESTIONABLE = 3
    BAD = 4
    NOT_REPORTED = 5
    INTERPOLATED = 6
    DESPIKED = 7
    NOT_SAMPLED = 9

    def __init__(self, flag):
        self.flag = flag

    @property
    def definition(self):
        defs = {
            0: "No Flag assigned",
            1: "Not calibrated.",
            2: "Acceptable measurement.",
            3: "Questionable measurement.",
            4: "Bad measurement.",
            5: "Not reported.",
            6: "Interpolated over a pressure interval larger than 2 dbar.",
            7: "Despiked.",
            9: "Not sampled.",
        }
        return defs[self.flag]

    @property
    def has_value(self):
        if self.flag in [5, 9]:
            return False
        return True


class ArgoFlag(IntEnum):
    GOOD = 1


class ExchangeFlagSample(IntEnum):
    NOFLAG = 0  # no idea if this will cause issue
    MISSING = 1
    GOOD = 2
    QUESTIONABLE = 3
    BAD = 4
    NOT_REPORTED = 5
    MEAN = 6
    CHROMA_MANUAL = 7
    CHROMA_IRREGULAR = 8
    NOT_SAMPLED = 9

    def __init__(self, flag):
        self.flag = flag

    @property
    def definition(self):
        defs = {
            0: "No Flag assigned",
            1: "Sample for this measurement was drawn from water bottle but analysis not received.",
            2: "Acceptable measurement.",
            3: "Questionable measurement.",
            4: "Bad measurement.",
            5: "Not reported.",
            6: "Mean of replicate measurements",
            7: "Manual chromatographic peak measurement.",
            8: "Irregular digital chromatographic peak integration.",
            9: "Sample not drawn for this measurement from this bottle.",
        }
        return defs[self.flag]

    def to_other(self, other_flag):
        if other_flag.__name__ == "ArgoFlag":
            mapping = {1: 1}
            # import ArgoFlag
            return None

    @classmethod
    def from_other(cls, other):
        return cls(0)

    @property
    def has_value(self):
        if self.flag in (1, 5, 9):
            return False
        return True


class ExchangeFlagBottle(IntEnum):
    NOFLAG = 0  # no idea if this will cause issue
    NO_INFO = 1
    GOOD = 2
    LEAKING = 3
    BAD_TRIP = 4
    NOT_REPORTED = 5
    DISCREPANCY = 6
    UNKNOWN = 7
    PAIR = 8
    NOT_SAMPLED = 9

    def __init__(self, flag):
        self.flag = flag

    @property
    def definition(self):
        defs = {
            0: "No Flag assigned",
            1: "Bottle information unavailable.",
            2: "No problems noted.",
            3: "Leaking.",
            4: "Did not trip correctly.",
            5: "Not reported.",
            6: "Significant discrepancy in measured values between Gerard and Niskin bottles.",
            7: "Unknown problem.",
            8: "Pair did not trip correctly. Note that the Niskin bottle can trip at an unplanned depth while the Gerard trips correctly and vice versa.",
            9: "Samples not drawn from this bottle.",
        }
        return defs[self.flag]

    @property
    def has_value(self):
        if self.flag in [1, 5, 9]:
            return False
        return True
