from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union
from datetime import date, time, datetime
from enum import Enum, auto
import io
from zipfile import is_zipfile

import logging

import requests

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

fmat = logging.Formatter(
    "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(fmat)

log.addHandler(ch)


exchange_doc = "https://exchange-format.readthedocs.io/en/latest"

ERRORS = {
    "utf8": f"Exchange files MUST be utf8 encoded: {exchange_doc}/common.html#encoding",
    "bom": f"Exchange files MUST NOT have a byte order mark: {exchange_doc}/common.html#byte-order-marks",  # noqa: E501
    "line-end": f"Exchange files MUST use LF line endings: {exchange_doc}/common.html#line-endings",  # noqa: E501
}


class InvalidExchangeFileError(ValueError):
    pass


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
        print("hi")
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
    expocode: str
    station: str
    cast: int
    sample: str  # may be the pressure value for CTD data

    @property
    def profile_id(self):
        return (self.expocode, self.station, self.cast)


@dataclass(frozen=True)
class ExchangeTimestamp(ValidateInitTypes, ToAndFromDict):
    date_part: date
    time_part: Union[time, None]

    def __init__(
        self, date_part: Union[str, date], time_part: Union[str, time, None] = None
    ):
        """This class is designed to be called using the "string" values normally
        found in an exchange file. This means it is looking for a date
        which will match "%Y%m%d" and a date which looks like "%H%M".
        The date may also be `None`

        """
        if isinstance(date_part, str):
            date_part = datetime.strptime(date_part, "%Y%m%d").date()
        object.__setattr__(self, "date_part", date_part)

        if isinstance(time_part, str):
            time_part = datetime.strptime(time_part, "%H%M").time()
        object.__setattr__(self, "time_part", time_part)

        self.__post_init__()


class FileType(Enum):
    CTD = auto()
    BOTTLE = auto()


@dataclass(frozen=True)
class Exchange:
    file_type: FileType
    comments: str


def read_exchange(filename_or_obj: Union[str, Path, io.BufferedIOBase]) -> Exchange:
    """Open an exchange file"""

    if isinstance(filename_or_obj, str) and filename_or_obj.startswith("http"):
        data_raw = io.BytesIO(requests.get(filename_or_obj).content)

    elif isinstance(filename_or_obj, (str, Path)):
        with open(filename_or_obj, "rb") as f:
            data_raw = io.BytesIO(f.read())

    elif isinstance(filename_or_obj, io.BufferedIOBase):
        data_raw = io.BytesIO(filename_or_obj.read())

    if is_zipfile(data_raw):
        raise NotImplementedError("zip files not supported yet")
    data_raw.seek(0)

    try:
        data = data_raw.read().decode("utf8")
    except UnicodeDecodeError as error:
        raise InvalidExchangeFileError(ERRORS["utf8"]) from error

    if data.startswith("\ufeff"):
        raise InvalidExchangeFileError(ERRORS["bom"])

    if "\r" in data:
        raise InvalidExchangeFileError(ERRORS["line-end"])

    return data
