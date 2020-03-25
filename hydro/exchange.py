from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from collections.abc import Mapping
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

from hydro.data import WHPNames, WHPName
from hydro.flag import ExchangeBottleFlag, ExchangeSampleFlag, ExchangeCTDFlag
from hydro.exceptions import (
    ExchangeEncodingError,
    ExchangeBOMError,
    ExchangeLEError,
    ExchangeMagicNumberError,
    ExchangeEndDataError,
    ExchangeDuplicateParameterError,
    ExchangeOrphanFlagError,
    ExchangeFlaglessParameterError,
    ExchangeParameterUnitAlignmentError,
    ExchangeDataColumnAlignmentError,
    ExchangeParameterUndefError,
    ExchangeFlagUnitError,
    ExchangeDataFlagPairError,
    ExchangeDataPartialKeyError,
    ExchangeDuplicateKeyError,
    ExchangeDataPartialCoordinateError,
    ExchangeDataInconsistentCoordinateError,
)

WHPNameIndex = Dict[WHPName, int]
ExchangeFlags = Union[ExchangeBottleFlag, ExchangeSampleFlag, ExchangeCTDFlag, None]


PROFILE_LEVEL_PARAMS = list(filter(lambda x: x.scope == "profile", WHPNames.values()))


class IntermediateDataPoint(NamedTuple):
    data: str
    flag: Optional[str]
    error: Optional[str]


def _bottle_get_params(
    params_units: Iterable[Tuple[str, Optional[str]]]
) -> WHPNameIndex:
    params = {}
    for index, (param, unit) in enumerate(params_units):
        if param in WHPNames.error_cols:
            continue
        if param.endswith("_FLAG_W"):
            continue
        try:
            params[WHPNames[(param, unit)]] = index
        except KeyError as error:
            raise ExchangeParameterUndefError(
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
            raise ExchangeFlagUnitError

        data_col = param.replace("_FLAG_W", "")
        try:
            whpname = whp_params_names[data_col]
            if whpname.flag_w is None:
                raise ExchangeFlaglessParameterError(f"{data_col}")
            param_flags[whpname] = index
        except KeyError as error:
            # we might have an alias...
            for name in whp_params:
                potential = [k[0] for k, v in WHPNames.items() if v == name]
                if data_col in potential:
                    param_flags[name] = index
                    break
            else:
                raise ExchangeOrphanFlagError(f"{data_col}") from error

    return param_flags


def _bottle_get_errors(
    params_units: Iterable[Tuple[str, Optional[str]]], whp_params: WHPNameIndex
) -> WHPNameIndex:
    param_errs = {}

    for index, (param, _unit) in enumerate(params_units):
        if param not in WHPNames.error_cols:
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
            if self.flag.has_value:
                msg = f"{self.whpname.whp_name} has a fill value but a flag of {self.flag}"
            else:
                msg = f"{self.whpname.whp_name} has the value {self.value} but a flag of {self.flag}"
            raise ExchangeDataFlagPairError(msg)


@dataclass(frozen=True)
class ExchangeCompositeKey(Mapping):
    expocode: str
    station: str
    cast: int
    sample: str  # may be the pressure value for CTD data
    _mapping: dict = field(init=False, repr=False, compare=False)

    EXPOCODE = WHPNames["EXPOCODE"]
    STNNBR = WHPNames["STNNBR"]
    CASTNO = WHPNames["CASTNO"]
    SAMPNO = WHPNames["SAMPNO"]

    WHP_PARAMS = (
        EXPOCODE,
        STNNBR,
        CASTNO,
        SAMPNO,
    )

    def __post_init__(self):
        object.__setattr__(
            self,
            "_mapping",
            {
                self.EXPOCODE: self.expocode,
                self.STNNBR: self.station,
                self.CASTNO: self.cast,
                self.SAMPNO: self.sample,
            },
        )

    @property
    def profile_id(self):
        return (self.expocode, self.station, self.cast)

    @classmethod
    def from_data_line(
        cls, data_line: Dict[WHPName, IntermediateDataPoint]
    ) -> ExchangeCompositeKey:
        EXPOCODE = cls.EXPOCODE
        STNNBR = cls.STNNBR
        CASTNO = cls.CASTNO
        SAMPNO = cls.SAMPNO
        return cls(
            expocode=EXPOCODE.data_type(data_line.pop(EXPOCODE).data),
            station=STNNBR.data_type(data_line.pop(STNNBR).data),
            cast=CASTNO.data_type(data_line.pop(CASTNO).data),
            sample=SAMPNO.data_type(data_line.pop(SAMPNO).data),
        )

    def __getitem__(self, key):
        return self._mapping[key]

    def __iter__(self):
        for key in self._mapping:
            yield key

    def __len__(self):
        return len(self._mapping)


@dataclass(frozen=True)
class ExchangeXYZT(Mapping):
    x: ExchangeDataPoint  # Longitude
    y: ExchangeDataPoint  # Latitude
    z: ExchangeDataPoint  # Pressure
    t: ExchangeTimestamp  # Time obviously...
    _mapping: dict = field(init=False, repr=False, compare=False)

    CTDPRS = WHPNames[("CTDPRS", "DBAR")]
    DATE = WHPNames[("DATE", None)]
    TIME = WHPNames[("TIME", None)]
    LATITUDE = WHPNames[("LATITUDE", None)]
    LONGITUDE = WHPNames[("LONGITUDE", None)]

    WHP_PARAMS: tuple = (
        CTDPRS,
        DATE,
        TIME,
        LATITUDE,
        LONGITUDE,
    )

    @classmethod
    def from_data_line(
        cls, data_line: Dict[WHPName, IntermediateDataPoint]
    ) -> ExchangeXYZT:

        date = data_line.pop(cls.DATE).data
        time: Optional[str] = None
        try:
            time = data_line.pop(cls.TIME).data
        except KeyError:
            pass

        return cls(
            x=ExchangeDataPoint.from_ir(cls.LONGITUDE, data_line.pop(cls.LONGITUDE)),
            y=ExchangeDataPoint.from_ir(cls.LATITUDE, data_line.pop(cls.LATITUDE)),
            z=ExchangeDataPoint.from_ir(cls.CTDPRS, data_line[cls.CTDPRS]),
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
            raise ExchangeDataPartialCoordinateError

        object.__setattr__(
            self,
            "_mapping",
            {
                self.LONGITUDE: self.x.value,
                self.LATITUDE: self.y.value,
                self.CTDPRS: self.z.value,
                self.TIME: self.t.time_part,
                self.DATE: self.t.date_part,
            },
        )

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
        return self._mapping[key]

    def __iter__(self):
        for key in self._mapping:
            yield key

    def __len__(self):
        return len(self._mapping)


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


class ExchangeDataProxy(Mapping):
    def __init__(self, exchange: Exchange):
        self._ex = exchange

    def __getitem__(self, key: Tuple[ExchangeCompositeKey, WHPName]):
        row, col = key
        if col in ExchangeCompositeKey.WHP_PARAMS:
            return row[col]
        elif col in ExchangeXYZT.WHP_PARAMS:
            return self._ex.coordinates[row][col]
        else:
            try:
                return self._ex.data[row][col].value
            except KeyError:
                return None

    def __iter__(self):
        for key in self._ex.keys:
            for param in self._ex.parameters:
                yield (key, param)

    def __len__(self):
        return len(self._ex.keys) * len(self._ex.parameters)


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

        # Check to see that all the "profile level" parameters are the same for
        # excah profile
        for key, group in groupby(self.keys, lambda k: k.profile_id):
            first_row = next(group)
            for col in PROFILE_LEVEL_PARAMS:
                val = self.at[(first_row, col)]
                for row in group:
                    if val != self.at[(row, col)]:
                        raise ExchangeDataInconsistentCoordinateError

    def __repr__(self):
        return f"""<hydro.Exchange profiles={len(self)}>"""

    def __len__(self):
        return len({key.profile_id for key in self.keys})

    @property
    def at(self) -> ExchangeDataProxy:
        return ExchangeDataProxy(self)

    def iter_profiles(self):
        for _key, group in groupby(self.keys, lambda k: k.profile_id):
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

    def flag_column_to_ndarray(self, col: WHPName) -> np.ndarray:
        if col not in self.flags:
            raise KeyError(f"No flags for {col}")

        a = []
        for key in self.keys:
            try:
                a.append(self.data[key][col].flag)
            except KeyError:
                a.append(None)

        return np.array(a, dtype=np.float)

    def column_to_ndarray(self, col: WHPName) -> np.ndarray:
        a = []
        dtype = col.data_type  # type: ignore
        for key in self.keys:
            a.append(self.at[(key, col)])

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
        import xarray as xr

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

            if var in self.flags:
                data_vars[f"var{n}_qc"] = np.full_like(data, np.nan, dtype=np.float)
            if var in self.errors:
                data_vars[f"var{n}_error"] = np.full_like(data, np.nan)

        for n_prof, prof in enumerate(self.iter_profiles()):
            for p_int, param in enumerate(prof.parameters):
                d = prof.column_to_ndarray(col=param)
                if param in one_d_vars:
                    data_vars[f"var{p_int}"][n_prof] = d[0]
                else:
                    data_vars[f"var{p_int}"][n_prof][: len(d)] = d

                if param in prof.flags:
                    d = prof.flag_column_to_ndarray(param)
                    data_vars[f"var{p_int}_qc"][n_prof][: len(d)] = d

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
            if v.endswith("_qc"):
                dataset[v].encoding["dtype"] = "int8"
                dataset[v].encoding["_FillValue"] = 9
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
    """Open an exchange file and return an :class:`hydro.exchange.Exchange` object
    """

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
        raise ExchangeEncodingError from error

    if data.startswith("\ufeff"):
        raise ExchangeBOMError

    if "\r" in data:
        raise ExchangeLEError

    if data.startswith("BOTTLE"):
        ftype = FileType.BOTTLE
    elif data.startswith("CTD"):
        ftype = FileType.CTD
    else:
        raise ExchangeMagicNumberError

    data_lines = deque(data.splitlines())
    stamp = data_lines.popleft()

    if "END_DATA" not in data_lines:
        raise ExchangeEndDataError

    comments = "\n".join([stamp, _extract_comments(data_lines)])

    # Strip end_data
    data_lines.remove("END_DATA")

    params = data_lines.popleft().split(",")
    # we can have a bunch of empty strings as units, we want these to be
    # None to match what would be in a WHPName object
    units = [x if x != "" else None for x in data_lines.popleft().split(",")]

    # at this point the data_lines should ONLY contain data/flags

    # column labels must be unique
    if len(params) != len(set(params)):
        raise ExchangeDuplicateParameterError

    # the number of expected columns is just going to be the number of
    # parameter names we see
    column_count = len(params)

    if len(units) != column_count:
        raise ExchangeParameterUnitAlignmentError

    whp_params = _bottle_get_params(zip(params, units))
    whp_flags = _bottle_get_flags(zip(params, units), whp_params)
    whp_errors = _bottle_get_errors(zip(params, units), whp_params)

    # ensure we will read the ENTIRE file
    if {*whp_params.values(), *whp_flags.values(), *whp_errors.values()} != set(
        range(column_count)
    ):
        raise RuntimeError(
            (
                "Not all of the data columns will be read. "
                "This shouldn't happen and is likely a bug, please include the file that caused "
                "this error to occur."
            )
        )

    line_parser = _bottle_line_parser(whp_params, whp_flags, whp_errors)

    exchange_data: Dict[ExchangeCompositeKey, dict] = {}
    coordinates = {}
    for data_line in data_lines:
        cols = [x.strip() for x in data_line.split(",")]
        if len(cols) != column_count:
            raise ExchangeDataColumnAlignmentError
        parsed_data_line = line_parser(data_line)
        try:
            key = ExchangeCompositeKey.from_data_line(parsed_data_line)
        except KeyError as error:
            raise ExchangeDataPartialKeyError from error

        try:
            coord = ExchangeXYZT.from_data_line(parsed_data_line)
        except KeyError as error:
            raise ExchangeDataPartialCoordinateError from error

        row_data = {
            param: ExchangeDataPoint.from_ir(param, ir)
            for param, ir in parsed_data_line.items()
        }
        if key in exchange_data:
            raise ExchangeDuplicateKeyError(f"{key}")

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
