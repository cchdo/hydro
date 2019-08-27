from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
from typing import Union, Iterable, Tuple, Optional, Mapping, Callable, List
from datetime import date, time, datetime
from enum import Enum, auto
import io
from zipfile import is_zipfile
from operator import itemgetter
from types import MappingProxyType

import requests

from hydro.data import WHPNames, WHPName
from hydro.flag import ExchangeBottleFlag, ExchangeSampleFlag, ExchangeCTDFlag

WHPNameIndex = Mapping[WHPName, int]
ExchangeFlags = Union[ExchangeBottleFlag, ExchangeSampleFlag, ExchangeCTDFlag]


exchange_doc = "https://exchange-format.readthedocs.io/en/latest"

ERRORS = {
    "utf8": f"Exchange files MUST be utf8 encoded: {exchange_doc}/common.html#encoding",
    "bom": f"Exchange files MUST NOT have a byte order mark: {exchange_doc}/common.html#byte-order-marks",  # noqa: E501
    "line-end": f"Exchange files MUST use LF line endings: {exchange_doc}/common.html#line-endings",  # noqa: E501
}


class InvalidExchangeFileError(ValueError):
    pass


class ToAndFromDict:
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _bottle_get_params(
    params_units: Iterable[Tuple[str, Optional[str]]]
) -> WHPNameIndex:
    params = {}
    for index, (param, unit) in enumerate(params_units):
        if param.endswith("_FLAG_W"):
            continue
        try:
            params[WHPNames[(param, unit)]] = index
        except KeyError as error:
            raise InvalidExchangeFileError(
                f"missing parameter def {(param, unit)}"
            ) from error
    return params


def _bottle_get_flags(
    params_units: Iterable[Tuple[str, Optional[str]]], whp_params: WHPNameIndex
) -> WHPNameIndex:

    param_flags = {}
    whp_params_names = {x.whp_name: x for x in whp_params.keys()}

    for index, (param, unit) in enumerate(params_units):
        if not param.endswith("_FLAG_W"):
            continue

        if unit is not None:
            raise InvalidExchangeFileError("Flags should not have units")

        try:
            whpname = whp_params_names[param.replace("_FLAG_W", "")]
            param_flags[whpname] = index
        except KeyError as error:
            raise InvalidExchangeFileError("Flag with no data column") from error

    return param_flags


def _bottle_make_readers(names_index: WHPNameIndex, flags_index: WHPNameIndex):
    _special_cases = [
        WHPNames[name]
        for name in [
            ("EXPOCODE", None),
            ("STNNBR", None),
            ("CASTNO", None),
            ("SAMPNO", None),
            ("DATE", None),
            ("TIME", None),
        ]
    ]

    for name in names_index:
        if name in _special_cases:
            continue

        # WHPName can have one of three data types:


@dataclass(frozen=True)
class ExchangeDataPoint:
    whpname: WHPName
    value: Union[str, float, int]
    flag: ExchangeFlags

    def __post_init__(self):
        # Check to see if the flag value allowes for data
        # Check to see if datatype is ok
        ...


@dataclass(frozen=True)
class ExchangeCompositeKey(ToAndFromDict):
    expocode: str
    station: str
    cast: int
    sample: str  # may be the pressure value for CTD data

    @property
    def profile_id(self):
        return (self.expocode, self.station, self.cast)

    @classmethod
    def key_factory(
        cls, params: WHPNameIndex
    ) -> Callable[[List[str]], ExchangeCompositeKey]:
        # Why is all this "hard coded"? These are the required ID keys according to the
        # spec, as such, their lack of presence is an error.
        try:
            expocode_index = params[WHPNames[("EXPOCODE", None)]]
            station_index = params[WHPNames[("STNNBR", None)]]
            cast_index = params[WHPNames[("CASTNO", None)]]
            sample_index = params[WHPNames[("SAMPNO", None)]]
        except KeyError as error:
            # TODO Message
            raise InvalidExchangeFileError("key getter") from error

        def index_getter(data_line: List[str]) -> ExchangeCompositeKey:
            return cls(
                expocode=str(itemgetter(expocode_index)(data_line)).strip(),
                station=str(itemgetter(station_index)(data_line)).strip(),
                cast=int(itemgetter(cast_index)(data_line)),
                sample=str(itemgetter(sample_index)(data_line)).strip(),
            )

        return index_getter


@dataclass(frozen=True)
class ExchangeTimestamp(ToAndFromDict):
    date_part: date
    time_part: Optional[time]

    @classmethod
    def from_strs(
        cls, date_part: str, time_part: Optional[str] = None
    ) -> ExchangeTimestamp:
        parsed_date = datetime.strptime(date_part, "%Y%m%d").date()
        parsed_time: Optional[time] = None
        if time_part is not None:
            parsed_time = datetime.strptime(time_part, "%H%M").time()
        return cls(date_part=parsed_date, time_part=parsed_time)

    @classmethod
    def key_factory(
        cls, params: WHPNameIndex
    ) -> Callable[[List[str]], ExchangeTimestamp]:
        time_index: Optional[int] = None
        try:
            date_index = params[WHPNames[("DATE", None)]]
        except KeyError as error:
            # TODO Message
            raise InvalidExchangeFileError("date key") from error
        try:
            time_index = params[WHPNames[("TIME", None)]]
        except KeyError:
            pass

        def index_getter(data_line: List[str]) -> ExchangeTimestamp:
            date_part = str(itemgetter(date_index)(data_line)).strip()
            time_part: Optional[str] = None

            if time_index is not None:
                time_part = str(itemgetter(time_index)(data_line)).strip()

            return cls.from_strs(date_part=date_part, time_part=time_part)

        return index_getter


class FileType(Enum):
    CTD = auto()
    BOTTLE = auto()


@dataclass(frozen=True)
class Exchange:
    file_type: FileType
    comments: str
    parameters: Tuple[WHPName, ...]
    keys: Tuple[ExchangeCompositeKey, ...]
    data: Mapping[ExchangeCompositeKey, Mapping[WHPName, Union[ExchangeTimestamp, str]]]

    def __post_init__(self):
        for key, value in self.data.items():
            self.data[key] = MappingProxyType(value)
        object.__setattr__(self, "data", MappingProxyType(self.data))

    def __repr__(self):
        return f"""<hydro.Exchange>"""


def _extract_comments(data: deque, include_post_content: bool = True) -> str:
    comments = []
    while data[0].startswith("#"):
        comments.append(data.popleft().lstrip("#"))

    if include_post_content:
        post_content = []
        while data[-1] != "END_DATA":
            post_content.append(data.pop())
        comments.extend(reversed(post_content))

    return "\n".join(comments)


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
    data_raw.seek(0)  # is_zipfile moves the "tell" position

    try:
        data = data_raw.read().decode("utf8")
    except UnicodeDecodeError as error:
        raise InvalidExchangeFileError(ERRORS["utf8"]) from error

    if data.startswith("\ufeff"):
        raise InvalidExchangeFileError(ERRORS["bom"])

    if "\r" in data:
        raise InvalidExchangeFileError(ERRORS["line-end"])

    if data.startswith("BOTTLE"):
        ftype = FileType.BOTTLE
    elif data.startswith("CTD"):
        ftype = FileType.CTD
    else:
        # TODO make messages
        raise InvalidExchangeFileError("message")

    data_lines = deque(data.splitlines())
    data_lines.popleft()  # discard "stamp"

    if "END_DATA" not in data_lines:
        # TODO make messages
        raise InvalidExchangeFileError("message")

    comments = _extract_comments(data_lines)

    # Strip end_data
    data_lines.remove("END_DATA")

    params = data_lines.popleft().split(",")
    # we can have a bunch of empty strings as units, we want these to be
    # None to match what would be in a WHPName object
    units = [x if x != "" else None for x in data_lines.popleft().split(",")]

    # at this point the data_lines should ONLY contain data/flags

    # column labels must be unique
    if len(params) != len(set(params)):
        # TODO Make Message
        raise InvalidExchangeFileError()

    # the number of expected columns is just going to be the number of
    # parameter names we see
    column_count = len(params)

    if len(units) != column_count:
        # TODO make message
        raise InvalidExchangeFileError()

    whp_params = _bottle_get_params(zip(params, units))
    whp_flags = _bottle_get_flags(zip(params, units), whp_params)

    # ensure we will read the ENTIRE file
    if {*whp_params.values(), *whp_flags.values()} != set(range(column_count)):
        raise ValueError("WAT")

    index_getter = ExchangeCompositeKey.key_factory(whp_params)
    datetime_getter = ExchangeTimestamp.key_factory(whp_params)
    # data_getters = _bottle_make_readers(whp_params, whp_flags)

    indicies = []
    exchange_data = {}
    for data_line in data_lines:
        cols = [x.strip() for x in data_line.split(",")]
        if len(cols) != column_count:
            raise InvalidExchangeFileError()
        index = index_getter(cols)
        indicies.append(index)
        exchange_data[index] = {
            param: datetime_getter(cols) for param in whp_params.keys()
        }

    return Exchange(
        file_type=ftype,
        comments=comments,
        parameters=tuple(whp_params.keys()),
        keys=tuple(indicies),
        data=exchange_data,
    )
