from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Union, Iterable, Tuple, Optional, Callable, Dict, NamedTuple
from datetime import date, time, datetime
from enum import Enum, auto
import io
from zipfile import is_zipfile
from operator import itemgetter
from itertools import groupby

import requests
import numpy as np
import xarray as xr

from hydro.data import WHPNames, WHPName
from hydro.flag import ExchangeBottleFlag, ExchangeSampleFlag, ExchangeCTDFlag

WHPNameIndex = Dict[WHPName, int]
ExchangeFlags = Union[ExchangeBottleFlag, ExchangeSampleFlag, ExchangeCTDFlag, None]

# Some Names we use frequently...
EXPOCODE = WHPNames[("EXPOCODE", None)]
STNNBR = WHPNames[("STNNBR", None)]
CASTNO = WHPNames[("CASTNO", None)]
SAMPNO = WHPNames[("SAMPNO", None)]
CTDPRS = WHPNames[("CTDPRS", "DBAR")]
DATE = WHPNames[("DATE", None)]
TIME = WHPNames[("TIME", None)]
LATITUDE = WHPNames[("LATITUDE", None)]
LONGITUDE = WHPNames[("LONGITUDE", None)]

WHP_ERROR_COLS = {
    ex.error_name: ex for ex in WHPNames.values() if ex.error_name is not None
}

EXCHANGE_COMPOSITE_KEY_PARAMS = (EXPOCODE, STNNBR, CASTNO, SAMPNO)

exchange_doc = "https://exchange-format.readthedocs.io/en/latest"

ERRORS = {
    "utf8": f"Exchange files MUST be utf8 encoded: {exchange_doc}/common.html#encoding",
    "bom": f"Exchange files MUST NOT have a byte order mark: {exchange_doc}/common.html#byte-order-marks",  # noqa: E501
    "line-end": f"Exchange files MUST use LF line endings: {exchange_doc}/common.html#line-endings",  # noqa: E501
}


class IntermediateDataPoint(NamedTuple):
    data: str
    flag: Optional[str]
    error: Optional[str]


class InvalidExchangeFileError(ValueError):
    pass


def _bottle_get_params(
    params_units: Iterable[Tuple[str, Optional[str]]]
) -> WHPNameIndex:
    params = {}
    for index, (param, unit) in enumerate(params_units):
        if param in WHP_ERROR_COLS:
            continue
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
            if whpname.flag_w is None:
                raise InvalidExchangeFileError(f"{whpname} cannot have a flag column")
            param_flags[whpname] = index
        except KeyError as error:
            raise InvalidExchangeFileError("Flag with no data column") from error

    return param_flags


def _bottle_get_errors(
    params_units: Iterable[Tuple[str, Optional[str]]], whp_params: WHPNameIndex
) -> WHPNameIndex:
    param_errs = {}

    for index, (param, unit) in enumerate(params_units):
        if param not in WHP_ERROR_COLS:
            continue

        for name in whp_params.keys():
            if name.error_name == param:
                param_errs[name] = index

    return param_errs


@dataclass(frozen=True)
class ExchangeDataPoint:
    whpname: WHPName
    value: Optional[Union[str, float, int]]
    error: Optional[float]
    flag: ExchangeFlags

    @classmethod
    def from_ir(cls, whpname: WHPName, ir: IntermediateDataPoint) -> ExchangeDataPoint:
        if ir.data.startswith("-999"):
            value = None
        else:
            # https://github.com/python/mypy/issues/5485
            value = whpname.data_type(ir.data)  # type: ignore

        flag: ExchangeFlags = None
        try:
            # we will catch the type error explicitly
            flag_v = int(ir.flag)  # type: ignore
            if whpname.flag_w == "woce_bottle":
                flag = ExchangeBottleFlag(flag_v)
            if whpname.flag_w == "woce_discrete":
                flag = ExchangeSampleFlag(flag_v)
            if whpname.flag_w == "woce_ctd":
                flag = ExchangeCTDFlag(flag_v)
        except TypeError:
            pass

        error: Optional[float] = None
        try:
            error = float(ir.error)  # type: ignore
        except TypeError:
            pass

        return ExchangeDataPoint(whpname=whpname, value=value, flag=flag, error=error)

    def __post_init__(self):
        if self.flag is not None and self.flag.has_value and self.value is None:
            raise InvalidExchangeFileError(f"{self}")


@dataclass(frozen=True)
class ExchangeCompositeKey:
    expocode: str
    station: str
    cast: int
    sample: str  # may be the pressure value for CTD data

    @property
    def profile_id(self):
        return (self.expocode, self.station, self.cast)

    @classmethod
    def from_data_line(
        cls, data_line: Dict[WHPName, IntermediateDataPoint]
    ) -> ExchangeCompositeKey:
        return cls(
            expocode=EXPOCODE.data_type(data_line.pop(EXPOCODE).data),
            station=STNNBR.data_type(data_line.pop(STNNBR).data),
            cast=CASTNO.data_type(data_line.pop(CASTNO).data),
            sample=SAMPNO.data_type(data_line.pop(SAMPNO).data),
        )

    def __getitem__(self, key):
        return {
            EXPOCODE: self.expocode,
            STNNBR: self.station,
            CASTNO: self.cast,
            SAMPNO: self.sample,
        }[key]


@dataclass(frozen=True)
class ExchangeXYZT:
    x: ExchangeDataPoint  # Longitude
    y: ExchangeDataPoint  # Latitude
    z: ExchangeDataPoint  # Pressure
    t: ExchangeTimestamp  # Time obviously...

    @classmethod
    def from_data_line(
        cls, data_line: Dict[WHPName, IntermediateDataPoint]
    ) -> ExchangeXYZT:
        date = data_line.pop(DATE).data
        time: Optional[str] = None
        try:
            time = data_line.pop(TIME).data
        except KeyError:
            pass

        return cls(
            x=ExchangeDataPoint.from_ir(LONGITUDE, data_line.pop(LONGITUDE)),
            y=ExchangeDataPoint.from_ir(LATITUDE, data_line.pop(LATITUDE)),
            z=ExchangeDataPoint.from_ir(CTDPRS, data_line[CTDPRS]),
            t=ExchangeTimestamp.from_strs(date, time),
        )

    def __repr__(self):
        return (
            f"<ExchangeXYZT "
            f"x={self.x.value} "
            f"y={self.y.value} "
            f"z={self.z.value} "
            f"t='{self.t.to_datetime}'>"
        )

    def __post_init__(self):
        if not all(
            [
                self.x.value is not None,
                self.y.value is not None,
                self.z.value is not None,
            ]
        ):
            raise InvalidExchangeFileError()

    def __eq__(self, other):
        return (self.x, self.y, self.z, self.t) == (other.x, other.y, other.z, other.t)

    def __lt__(self, other):
        """We will consider the following order:
        * A later coordiante is greater than an earlier one
        * A deeper coordinate is greater than a shallower one
        * A more easternly coordinate is greater than a more westerly one
        * A more northernly coordinate is greater than a more southerly one
        The first two points should get most of the stuff we care about sorted
        """
        return (self.t.to_datetime, self.z.value, self.x.value, self.y.value) < (
            other.t.to_datetime,
            other.z.value,
            other.x.value,
            other.y.value,
        )

    def __getitem__(self, key):
        return {
            LONGITUDE: self.x.value,
            LATITUDE: self.y.value,
            CTDPRS: self.z.value,
            TIME: self.t.time_part,
            DATE: self.t.date_part,
        }[key]


@dataclass(frozen=True)
class ExchangeTimestamp:
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

    @property
    def to_datetime(self):
        if self.time_part:
            return datetime.combine(self.date_part, self.time_part)
        else:
            return datetime.combine(self.date_part, time(0, 0))

    def __lt__(self, other):
        return self.to_datetime < other.to_datetime

    def __eq__(self, other):
        return self.to_datetime == other.to_datetime


def _bottle_line_parser(
    names_index: WHPNameIndex, flags_index: WHPNameIndex, errors_index: WHPNameIndex
) -> Callable[[str], Dict[WHPName, IntermediateDataPoint]]:
    data_getters = {}
    flag_getters = {}
    error_getters = {}
    for name, data_col in names_index.items():
        data_getter = itemgetter(data_col)
        flag_col = flags_index.get(name)
        error_col = errors_index.get(name)

        data_getters[name] = data_getter
        if flag_col is None:
            flag_getters[name] = lambda x: None
        else:
            flag_getters[name] = itemgetter(flag_col)

        if error_col is None:
            error_getters[name] = lambda x: None
        else:
            error_getters[name] = itemgetter(error_col)

    def line_parser(line: str) -> Dict[WHPName, IntermediateDataPoint]:
        split_line = [s.strip() for s in line.split(",")]
        parsed = {}
        for name in names_index:
            data: str = data_getters[name](split_line)
            flag: Optional[str] = flag_getters[name](split_line)
            error: Optional[str] = error_getters[name](split_line)

            parsed[name] = IntermediateDataPoint(data, flag, error=error)

        return parsed

    return line_parser


class FileType(Enum):
    CTD = auto()
    BOTTLE = auto()


@dataclass(frozen=True)
class Exchange:
    file_type: FileType
    comments: str
    parameters: Tuple[WHPName, ...]
    flags: Tuple[WHPName, ...]
    errors: Tuple[WHPName, ...]
    keys: Tuple[ExchangeCompositeKey, ...]
    coordinates: Dict[ExchangeCompositeKey, ExchangeXYZT]
    data: Dict[ExchangeCompositeKey, Dict[WHPName, ExchangeDataPoint]]

    def __post_init__(self):
        # first the keys are sorted by information contained in the coordinates
        sorted_keys = sorted(self.keys, key=lambda x: self.coordinates[x])

        # this checks to see if the number of unique profile_ids would be the same
        # lengths as the number of profiles we woudl get when "iter_profiles"
        if len({key.profile_id for key in sorted_keys}) != len(
            list(key for key in groupby(sorted_keys, lambda k: k.profile_id))
        ):
            # this probably means there was no time available (or it was all 0000)
            # so we need to sort by the profile_id
            sorted_keys = sorted(sorted_keys, key=lambda x: x.profile_id)
        object.__setattr__(self, "keys", tuple(sorted_keys))

    def __repr__(self):
        return f"""<hydro.Exchange profiles={len(self)}>"""

    def __len__(self):
        return len({key.profile_id for key in self.keys})

    def iter_profiles(self):
        for key, group in groupby(self.keys, lambda k: k.profile_id):
            keys = tuple(group)
            yield Exchange(
                file_type=self.file_type,
                comments=self.comments,
                parameters=self.parameters,
                flags=self.flags,
                errors=self.errors,
                keys=keys,
                coordinates={
                    sample_id: self.coordinates[sample_id] for sample_id in keys
                },
                data={sample_id: self.data[sample_id] for sample_id in keys},
            )

    def column_to_ndarray(self, col: WHPName) -> np.ndarray:
        a = []
        dtype = col.data_type  # type: ignore
        if col in EXCHANGE_COMPOSITE_KEY_PARAMS:
            for key in self.keys:
                a.append(key[col])

        elif col in (LATITUDE, LONGITUDE, CTDPRS, DATE, TIME):
            for key in self.keys:
                a.append(self.coordinates[key][col])
        else:
            for key in self.keys:
                try:
                    a.append(self.data[key][col].value)
                except KeyError:
                    a.append(None)

        if None not in a:  # contigious array, should just work
            return np.array(a, dtype=dtype)
        elif dtype is str:
            return np.array(["" if s is None else s for s in a], dtype=str)
        else:  # turn int arrays with none in one with NaNs
            return np.array(a, dtype=float)

    def iter_profile_coordinates(self):
        for profile in self.iter_profiles():
            yield profile.coordinates[profile.keys[-1]]

    def to_xarray(self):
        """
        Current thinking:
        There are a few "special case" variables which include the WHP identifing ones:
        * EXPOCODE
        * STNNBR
        * CASTNO
        * SAMPNO
        Profile level spacetime coords:
        * LATITUDE
        * LONGITUDE
        * DATE
        * TIME
        * CTDPRS
        If present, bottle trip information:
        * BTL_LAT
        * BTL_LON
        * BTL_DATE
        * BTL_TIME

        Note that the seperate date and time need to be combined into a single
        date var for CF. Except for the bottle trip information, all the
        above should probably get "real" var names not just var0, ..., varN.
        """

        N_PROF = len(self)
        N_LEVELS = max([len(prof.keys) for prof in self.iter_profiles()])
        one_d_vars = list(filter(lambda v: v.scope == "profile", WHPNames.values()))
        one_d_dims = {"N_PROF": N_PROF}
        data_vars = {}
        dims = {"N_PROF": N_PROF, "N_LEVELS": N_LEVELS}
        for n, var in enumerate(self.parameters):
            if var in one_d_vars:
                size = N_PROF
            else:
                size = (N_PROF, N_LEVELS)

            if var.data_type is str:  # type: ignore
                data = np.empty(size, dtype=object)
            else:
                data = np.zeros(size, dtype=float)
                data[:] = np.nan
            data_vars[f"var{n}"] = data
        for n_prof, prof in enumerate(self.iter_profiles()):
            for p_int, param in enumerate(prof.parameters):
                d = prof.column_to_ndarray(col=param)
                if param in one_d_vars:
                    data_vars[f"var{p_int}"][n_prof] = d[0]
                else:
                    data_vars[f"var{p_int}"][n_prof][: len(d)] = d

        dvars = {}
        for k, v in data_vars.items():
            if v.ndim == 1:
                dvars[k] = (one_d_dims, v)
            else:
                dvars[k] = (dims, v)
        # dvars = {k: (dims, v) for k, v in data_vars.items()}
        dataset = xr.Dataset(dvars)
        for v in dataset:
            if dataset[v].dtype == object:
                dataset[v].encoding["dtype"] = "str"
            if dataset[v].dtype == float:
                dataset[v].encoding["dtype"] = "float32"
        return dataset


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
    whp_errors = _bottle_get_errors(zip(params, units), whp_params)

    # ensure we will read the ENTIRE file
    if {*whp_params.values(), *whp_flags.values(), *whp_errors.values()} != set(
        range(column_count)
    ):
        raise ValueError("WAT")

    line_parser = _bottle_line_parser(whp_params, whp_flags, whp_errors)

    exchange_data = {}
    coordinates = {}
    for data_line in data_lines:
        cols = [x.strip() for x in data_line.split(",")]
        if len(cols) != column_count:
            raise InvalidExchangeFileError()
        parsed_data_line = line_parser(data_line)
        try:
            key = ExchangeCompositeKey.from_data_line(parsed_data_line)
        except KeyError as error:
            raise InvalidExchangeFileError("Something Missing") from error

        try:
            coord = ExchangeXYZT.from_data_line(parsed_data_line)
        except KeyError as error:
            raise InvalidExchangeFileError("Something Missing") from error

        row_data = {
            param: ExchangeDataPoint.from_ir(param, ir)
            for param, ir in parsed_data_line.items()
        }
        exchange_data[key] = dict(filter(lambda di: di[1].flag != 9, row_data.items()))
        coordinates[key] = coord

    return Exchange(
        file_type=ftype,
        comments=comments,
        parameters=tuple(whp_params.keys()),
        flags=tuple(whp_flags.keys()),
        errors=tuple(whp_errors.keys()),
        keys=tuple(exchange_data.keys()),
        coordinates=coordinates,
        data=exchange_data,
    )
