from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import (
    Union,
    Tuple,
    Optional,
    Dict,
    NamedTuple,
    IO,
)
from datetime import datetime, timezone
from enum import Enum
from itertools import groupby
from functools import cached_property
from zipfile import ZipFile, ZIP_DEFLATED
import warnings
import string
from functools import wraps
from logging import getLogger
import os  # noqa

import numpy as np
import pandas as pd
import xarray as xr

from cchdo.params import WHPNames, WHPName
from cchdo.params._version import version as params_version
from .flags import ExchangeBottleFlag, ExchangeSampleFlag, ExchangeCTDFlag
from .exceptions import (
    ExchangeDataFlagPairError,
    ExchangeDataPartialCoordinateError,
    ExchangeDataInconsistentCoordinateError,
)

try:
    from .. import __version__ as hydro_version

    CCHDO_VERSION = ".".join(hydro_version.split(".")[:2])
    if "dev" in hydro_version:
        CCHDO_VERSION = hydro_version
except ImportError:
    hydro_version = CCHDO_VERSION = "unknown"

log = getLogger(__name__)


WHPNameIndex = Dict[WHPName, int]
ExchangeFlags = Union[ExchangeBottleFlag, ExchangeSampleFlag, ExchangeCTDFlag, None]


PROFILE_LEVEL_PARAMS = list(filter(lambda x: x.scope == "profile", WHPNames.values()))

DIMS = ("N_PROF", "N_LEVELS")

FLAG_SCHEME = {
    "woce_bottle": ExchangeBottleFlag,
    "woce_discrete": ExchangeSampleFlag,
    "woce_ctd": ExchangeCTDFlag,
}


class IntermediateDataPoint(NamedTuple):
    data: str
    flag: Optional[str]
    error: Optional[str]


@dataclass(frozen=True)
class ExchangeDataPoint:
    whpname: WHPName
    value: Optional[Union[str, float, int]]
    error: Optional[float]
    flag: ExchangeFlags
    source_c_format: Optional[int] = field(compare=False)

    @classmethod
    def from_ir(cls, whpname: WHPName, ir: IntermediateDataPoint) -> ExchangeDataPoint:
        source_c_format = None
        if ir.data.startswith("-999"):
            value = None
        else:
            if len(splt := ir.data.split(".")) == 2:
                source_c_format = len(splt[-1])
            elif whpname.data_type is float:  # type: ignore
                source_c_format = 0
            # https://github.com/python/mypy/issues/5485
            value = whpname.data_type(ir.data)  # type: ignore

        flag: ExchangeFlags = None
        try:
            # we will catch the type error explicitly
            flag_v = int(ir.flag)  # type: ignore
            flag = FLAG_SCHEME[whpname.flag_w](flag_v)  # type: ignore
        except TypeError:
            pass

        error: Optional[float] = None
        if ir.error is not None and not ir.error.startswith("-999"):
            try:
                error = float(ir.error)  # type: ignore
            except TypeError:
                pass

        return ExchangeDataPoint(
            whpname=whpname,
            value=value,
            flag=flag,
            error=error,
            source_c_format=source_c_format,
        )

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
    t: np.datetime64  # Time obviously...
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

    TEMPORAL_PARAMS = (DATE, TIME)

    CF_AXIS = {
        DATE: "T",
        TIME: "T",
        LATITUDE: "Y",
        LONGITUDE: "X",
        CTDPRS: "Z",
    }

    @classmethod
    def from_data_line(
        cls, data_line: Dict[WHPName, IntermediateDataPoint]
    ) -> ExchangeXYZT:

        units = "D"
        date = datetime.strptime(data_line.pop(cls.DATE).data, "%Y%m%d")
        try:
            time = data_line.pop(cls.TIME).data
            time_obj = datetime.strptime(time, "%H%M").time()
            date = datetime.combine(date.date(), time_obj)
            units = "m"
        except KeyError:
            pass

        return cls(
            x=ExchangeDataPoint.from_ir(cls.LONGITUDE, data_line.pop(cls.LONGITUDE)),
            y=ExchangeDataPoint.from_ir(cls.LATITUDE, data_line.pop(cls.LATITUDE)),
            z=ExchangeDataPoint.from_ir(cls.CTDPRS, data_line[cls.CTDPRS]),
            t=np.datetime64(date, units),
        )

    def __repr__(self):
        return (
            f"<ExchangeXYZT "
            f"x={self.x.value} "
            f"y={self.y.value} "
            f"z={self.z.value} "
            f"t={self.t!r}>"
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
                self.TIME: self._time_part,
                self.DATE: self._date_part,
            },
        )

    @cached_property
    def _source_c_formats(self):
        return {
            self.LONGITUDE: self.x.source_c_format,
            self.LATITUDE: self.y.source_c_format,
            self.CTDPRS: self.z.source_c_format,
            self.TIME: None,
            self.DATE: None,
        }

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
        return (self.t, self.z.value, self.x.value, self.y.value) < (
            other.t,
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

    @property
    def _time_part(self):
        if self.t.dtype.name == "datetime64[D]":
            return None
        return pd.Timestamp(self.t).to_pydatetime().time()

    @property
    def _date_part(self):
        return pd.Timestamp(self.t).to_pydatetime().date()


def _ndcache_helper(key):
    def _ndcache_dec(func):
        @wraps(func)
        def _ndcache_func(self: Exchange, param: Optional[WHPName] = None):
            try:
                return self._ndarray_cache[(param, key)]
            except KeyError:
                if param is None:
                    self._ndarray_cache[(param, key)] = func(self)
                else:
                    self._ndarray_cache[(param, key)] = func(self, param)
                return self._ndarray_cache[(param, key)]

        return _ndcache_func

    return _ndcache_dec


class FileType(Enum):
    CTD = "C"
    BOTTLE = "B"


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

    _ndarray_cache: Dict[Tuple[Optional[WHPName], str], np.ndarray] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self):
        # first the keys are sorted by information contained in the coordinates
        sorted_keys = sorted(self.keys, key=lambda x: self.coordinates[x])

        # Make params ordered by preferred order (might change with params DB)
        object.__setattr__(self, "parameters", tuple(sorted(self.parameters)))

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

        for col in PROFILE_LEVEL_PARAMS:
            try:
                data = np.transpose(self.parameter_to_ndarray(col))
            except KeyError:
                continue

            if data.dtype == float:
                if not ((data == data[0]) | np.isnan(data)).all():
                    raise ExchangeDataInconsistentCoordinateError
            else:
                if not ((data == data[0]) | (data == "")).all():
                    raise ExchangeDataInconsistentCoordinateError

        # validate BTL_DATE and BTL_TIME if present...
        if (WHPNames["BTL_DATE"] in self.parameters) != (
            WHPNames["BTL_TIME"] in self.parameters
        ):
            raise ValueError("BTL_DATE or BTL_TIME present when the other is not")
        if WHPNames["BTL_DATE"] in self.parameters:
            for key, row in self.data.items():
                date = row.get(WHPNames["BTL_DATE"])
                time = row.get(WHPNames["BTL_TIME"])

                if (date.value is None) != (time.value is None):
                    raise ValueError("BTL_TIME or BTL_DATE have mismatched fill values")

                if date.value is None:
                    continue

                if len(time.value) != 4:
                    warnings.warn("Left padding BTL_TIME with zeros")
                    object.__setattr__(
                        self.data[key][WHPNames["BTL_TIME"]],
                        "value",
                        time.value.zfill(4),
                    )
            self.sampletime_to_ndarray()

    def __repr__(self):
        return f"""<hydro.Exchange profiles={len(self)}>"""

    def __len__(self):
        return self.shape[0]

    @cached_property
    def shape(self):
        x = len({key.profile_id for key in self.keys})
        y = max(
            [len(list(prof)) for _, prof in groupby(self.keys, lambda k: k.profile_id)]
        )
        return (x, y)

    @cached_property
    def _param_source_numeric_precision(self) -> Dict[WHPName, int]:
        c_formats_max: Dict[WHPName, int] = {}
        for coord in self.coordinates.values():
            for param, c_format in coord._source_c_formats.items():
                if c_format is None:
                    continue
                c_formats_max[param] = max(c_format, c_formats_max.get(param, 0))

        for _, row in self.data.items():
            for param, datum in row.items():
                if datum.source_c_format is not None:
                    c_formats_max[param] = max(
                        datum.source_c_format, c_formats_max.get(param, 0)
                    )

        return c_formats_max

    @cached_property
    def _param_source_c_formats(self) -> Dict[WHPName, str]:
        c_formats = {}
        for param, C_format in self._param_source_numeric_precision.items():
            c_formats[param] = f"%.{C_format}f"

        return c_formats

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

    @_ndcache_helper("flag")
    def flag_to_ndarray(self, param: WHPName) -> np.ndarray:
        if param not in self.flags:
            raise KeyError(f"No flags for {param}")

        arr = np.full(self.shape, np.nan, dtype=float)

        try:
            data = self._col_major_data[param]
        except KeyError:
            # this means there is no data and we can jsut use the empty array
            pass
        else:
            for key, value in data.items():
                idx = self.ndaray_indicies[key]
                arr[idx] = value.flag

        return arr

    def flag_to_dataarray(
        self, param: WHPName, name: Optional[str] = None
    ) -> xr.DataArray:
        data = self.flag_to_ndarray(param)

        dims = DIMS[: data.ndim]

        flag_defs = FLAG_SCHEME[param.flag_w]  # type: ignore
        flag_values = []
        flag_meanings = []
        for flag in flag_defs:
            flag_values.append(int(flag))
            flag_meanings.append(flag.cf_def)  # type: ignore

        odv_conventions_map = {
            "woce_bottle": "WOCESAMPLE - WOCE Quality Codes for the sampling device itself",
            "woce_ctd": "WOCECTD - WOCE Quality Codes for CTD instrument measurements",
            "woce_discrete": "WOCEBOTTLE - WOCE Quality Codes for water sample (bottle) measurements",
        }

        attrs = {
            "standard_name": "status_flag",
            "flag_values": np.array(flag_values, dtype="int8"),
            "flag_meanings": " ".join(flag_meanings),
            "conventions": odv_conventions_map[param.flag_w],  # type: ignore
        }

        da = xr.DataArray(data=data, dims=dims, attrs=attrs, name=name)

        da.encoding["dtype"] = "int8"
        da.encoding["_FillValue"] = 9

        return da

    @_ndcache_helper("error")
    def error_to_ndarray(self, param: WHPName) -> np.ndarray:
        if param not in self.errors:
            raise KeyError(f"No error for {param}")

        arr = np.full(self.shape, np.nan, dtype=float)

        try:
            data = self._col_major_data[param]
        except KeyError:
            # this means there is no data and we can jsut use the empty array
            pass
        else:
            for key, value in data.items():
                idx = self.ndaray_indicies[key]
                arr[idx] = value.error

        return arr

    def error_to_dataarray(
        self, param: WHPName, name: Optional[str] = None
    ) -> xr.DataArray:
        data = self.error_to_ndarray(param)
        dims = DIMS[: data.ndim]
        attrs = param.get_nc_attrs(error=True)

        da = xr.DataArray(data=data, dims=dims, attrs=attrs, name=name)
        da.encoding["dtype"] = "float64"

        return da

    @_ndcache_helper("time")
    def time_to_ndarray(self) -> np.ndarray:
        """Time is a specal/funky case

        .. todo::

            Write why time is specal in exchange
        """
        arr = np.full(self.shape, np.datetime64("NaT"), dtype="datetime64[m]")

        for row, (_key, group) in enumerate(groupby(self.keys, lambda k: k.profile_id)):
            for col, key in enumerate(group):
                arr[row, col] = self.coordinates[key].t

        return arr

    def time_precision(self) -> float:
        precisions = []
        for _key, group in groupby(self.keys, lambda k: k.profile_id):
            for key in group:
                if self.coordinates[key].t.dtype.name == "datetime64[D]":
                    precisions.append(1.0)
                else:
                    precisions.append(0.000694)  # minute fraction of a day
        return min(precisions)

    def time_to_dataarray(self) -> xr.DataArray:
        data = self.time_to_ndarray()[:, 0]
        # units will be handeled by xarray on serialization
        precision = self.time_precision()
        attrs = {
            "standard_name": "time",
            "axis": "T",
            "whp_name": ["DATE", "TIME"],
            "resolution": precision,
        }
        dims = DIMS[: data.ndim]
        da = xr.DataArray(
            name="time",
            data=data,
            dims=dims,
            attrs=attrs,
        )
        da.encoding["_FillValue"] = None
        da.encoding["units"] = "days since 1950-01-01T00:00Z"
        da.encoding["calendar"] = "gregorian"
        da.encoding["dtype"] = "double"
        return da

    @_ndcache_helper("date")
    def sampletime_to_ndarray(self) -> np.ndarray:
        arr = np.full(self.shape, np.datetime64("NaT"), dtype="datetime64[m]")

        for row, (_key, group) in enumerate(groupby(self.keys, lambda k: k.profile_id)):
            for col, key in enumerate(group):
                date = self.data.get(key, {}).get(WHPNames["BTL_DATE"])
                time = self.data.get(key, {}).get(WHPNames["BTL_TIME"])
                if (date is not None and date.value is not None) and (
                    time is not None and time.value is not None
                ):
                    arr[row, col] = datetime.strptime(
                        f"{date.value}{time.value}", "%Y%m%d%H%M"
                    )
                else:
                    arr[row, col] = None

        return arr

    def sampletime_to_dataarray(self) -> xr.DataArray:
        data = self.sampletime_to_ndarray()
        attrs = {
            "standard_name": "time",
            "whp_name": ["BTL_DATE", "BTL_TIME"],
            "resolution": 0.000694,
        }
        dims = DIMS[: data.ndim]
        da = xr.DataArray(
            name="bottle_time",
            data=data,
            dims=dims,
            attrs=attrs,
        )
        # TODO Write a migrator for the new fill value
        da.encoding["_FillValue"] = np.NaN
        da.encoding["units"] = "days since 1950-01-01T00:00Z"
        da.encoding["calendar"] = "gregorian"
        return da

    def coord_to_dataarray(self, param: WHPName) -> xr.DataArray:
        if (
            param not in ExchangeXYZT.WHP_PARAMS
            or param in ExchangeXYZT.TEMPORAL_PARAMS
        ):
            raise ValueError("param must be one of: LATITUDE, LONGITUDE, CTDPRS")

        axis_to_name = {"X": "longitude", "Y": "latitude", "Z": "pressure"}

        data = self.parameter_to_ndarray(param)

        if param.scope == "profile":
            data = data[:, 0]

        attrs = param.get_nc_attrs()
        if param in self._param_source_c_formats:
            attrs["source_C_format"] = self._param_source_c_formats[param]

        axis = ExchangeXYZT.CF_AXIS[param]
        attrs["axis"] = axis
        if axis == "Z":
            attrs["positive"] = "down"

        dims = DIMS[: data.ndim]
        name = axis_to_name[axis]

        da = xr.DataArray(
            name=name,
            data=data,
            dims=dims,
            attrs=attrs,
        )
        da.encoding["zlib"] = True

        if not np.any(np.isnan(data)):
            da.encoding["_FillValue"] = None

        if data.dtype == object:
            da.encoding["dtype"] = "S1"

        return da

    def key_to_dataarray(self, param: WHPName) -> xr.DataArray:
        if param not in ExchangeCompositeKey.WHP_PARAMS:
            raise ValueError(f"param must be one of: {ExchangeCompositeKey.WHP_PARAMS}")

        key_to_name = {
            ExchangeCompositeKey.EXPOCODE: "expocode",
            ExchangeCompositeKey.STNNBR: "station",
            ExchangeCompositeKey.CASTNO: "cast",
            ExchangeCompositeKey.SAMPNO: "sample",
        }

        data = self.parameter_to_ndarray(param)

        if param.scope == "profile":
            data = data[:, 0]

        if param.data_type == int:  # type: ignore
            data = data.astype(int)

        attrs = param.get_nc_attrs()

        dims = DIMS[: data.ndim]
        name = key_to_name[param]

        da = xr.DataArray(
            name=name,
            data=data,
            dims=dims,
            attrs=attrs,
        )

        da.encoding["zlib"] = True

        if param.data_type == int:  # type: ignore
            # the woce spec says this should go from 1 and incriment
            # largest I have seen is maybe 20something on a GT cruise
            da.encoding["dtype"] = "int8"

        if data.dtype == object:
            da.encoding["dtype"] = "S1"

        return da

    @cached_property
    def ndaray_indicies(self):
        profiles = groupby(self.keys, lambda k: k.profile_id)

        indicies = {}
        for row, (_key, levels) in enumerate(profiles):
            for col, key in enumerate(levels):
                indicies[key] = (row, col)

        return indicies

    @cached_property
    def _col_major_data(self):
        from collections import defaultdict

        data = defaultdict(dict)
        for key, row in self.data.items():
            for param, datum in row.items():
                data[param][key] = datum

        return dict(data)

    @_ndcache_helper("value")
    def parameter_to_ndarray(self, param: WHPName) -> np.ndarray:
        # https://github.com/python/mypy/issues/5485
        dtype = param.data_type  # type: ignore
        if dtype == str:
            arr = np.full(self.shape, "", dtype=object)
        else:
            arr = np.full(self.shape, np.nan, dtype=float)

        if param not in (*ExchangeCompositeKey.WHP_PARAMS, *ExchangeXYZT.WHP_PARAMS):
            try:
                data = self._col_major_data[param]
            except KeyError:
                # this means there is no data and we can jsut use the empty array
                pass
            else:
                for key, value in data.items():
                    idx = self.ndaray_indicies[key]
                    arr[idx] = value.value
        elif param in ExchangeXYZT.WHP_PARAMS:
            for key, value in self.coordinates.items():
                idx = self.ndaray_indicies[key]
                arr[idx] = value._mapping[param]
        else:
            for key in self.keys:
                idx = self.ndaray_indicies[key]
                arr[idx] = key._mapping[param]

        if dtype == str:
            arr[arr == None] = ""  # noqa

        return arr

    def parameter_to_dataarray(
        self, param: WHPName, name: Optional[str] = None
    ) -> xr.DataArray:
        data = self.parameter_to_ndarray(param)

        if param.scope == "profile":
            data = data[:, 0]

        dims = DIMS[: data.ndim]

        attrs = param.get_nc_attrs()
        if param in self._param_source_c_formats:
            attrs["source_C_format"] = self._param_source_c_formats[param]

        da = xr.DataArray(data=data, dims=dims, attrs=attrs, name=name)

        da.encoding["zlib"] = True

        if data.dtype == object:
            da.encoding["dtype"] = "S1"
        if data.dtype == float:
            da.encoding["dtype"] = "float64"

        return da

    def iter_profile_coordinates(self):
        for profile in self.iter_profiles():
            yield profile.coordinates[profile.keys[-1]]

    def to_xarray(self, source_c_format=True):
        consumed = []
        data_arrays = []
        coords = {}

        # coordinates
        for param in ExchangeXYZT.WHP_PARAMS:
            consumed.append(param)
            if param in ExchangeXYZT.TEMPORAL_PARAMS:
                continue
            coord = self.coord_to_dataarray(param)
            coords[coord.name] = coord

        # Time Special case
        temporal = self.time_to_dataarray()
        coords[temporal.name] = temporal

        if WHPNames["BTL_DATE"] in self.parameters:
            consumed.append(WHPNames["BTL_TIME"])
            consumed.append(WHPNames["BTL_DATE"])
            data_arrays.append(self.sampletime_to_dataarray())

        # CCHDO Sample Identifying parameters go into coords
        for param in ExchangeCompositeKey.WHP_PARAMS:
            consumed.append(param)
            coord = self.key_to_dataarray(param)
            coords[coord.name] = coord

        data_params = (param for param in self.parameters if param not in consumed)
        varN = 0
        for _, param in enumerate(data_params):
            if param.nc_name is not None:
                name = param.nc_name
            else:
                name = f"var{varN}"
                varN += 1
            da = self.parameter_to_dataarray(param, name=f"{name}")
            data_arrays.append(da)

            ancillary_variables = []

            if param in self.flags:
                da_qc = self.flag_to_dataarray(param, name=f"{name}_qc")
                data_arrays.append(da_qc)

                ancillary_variables.append(da_qc.name)

            if param in self.errors:
                da_error = self.error_to_dataarray(param, name=f"{name}_error")
                data_arrays.append(da_error)

                ancillary_variables.append(da_error.name)

            # analytical temperature linking
            # TODO: write a migrator/tests for this
            if param.analytical_temperature_name is not None:
                temperature_param = WHPNames[
                    (
                        param.analytical_temperature_name,
                        param.analytical_temperature_units,
                    )
                ]
                if temperature_param in self.parameters:
                    ancillary_variables.append(temperature_param.nc_name)

            if len(ancillary_variables) > 0:
                da.attrs["ancillary_variables"] = " ".join(ancillary_variables)

        # record the types of the profiles, this is probably "least" importnat so it can go at the end
        profile_type = xr.DataArray(
            [self.file_type.value] * len(self), name="profile_type", dims=DIMS[0]
        )
        profile_type.encoding[
            "dtype"
        ] = "S1"  # we probably always want this to be a char for max compatability
        data_arrays.append(profile_type)

        # CF 1.8 geometry container
        data_arrays.append(
            xr.DataArray(
                name="geometry_container",
                attrs={
                    "geometry_type": "point",
                    "node_coordinates": "longitude latitude",
                },
            )
        )

        dataset = xr.Dataset(
            {da.name: da for da in data_arrays},
            coords=coords,
            attrs={
                "Conventions": f"CF-1.8 CCHDO-{CCHDO_VERSION}",
                "cchdo_software_version": f"hydro {hydro_version}",
                "cchdo_parameters_version": f"params {params_version}",
                "comments": self.comments,
                "featureType": "profile",
            },
        )

        # Add geometry to useful vars
        for var in ("expocode", "station", "cast", "section_id", "time"):
            if var in dataset:
                dataset[var].attrs["geometry"] = "geometry_container"

        if not source_c_format:
            for variable in dataset.variables:
                da = dataset[variable]
                try:
                    del da.attrs["source_C_format"]
                except KeyError:
                    continue

        return dataset

    def gen_fname(self: Exchange) -> str:
        allowed_chars = set(f"._{string.ascii_letters}{string.digits}")

        key = self.keys[0]
        if self.file_type == FileType.BOTTLE:
            fname = f"{key.expocode}_hy1.csv"
        elif self.file_type == FileType.CTD and len(self) > 1:
            fname = f"{key.expocode}_ct1.zip"
        else:
            fname = f"{key.expocode}_{key.station}_{key.cast}_ct1.csv"

        for char in set(fname) - allowed_chars:
            fname = fname.replace(char, "_")

        return fname

    def to_exchange_csv(
        self,
        filename_or_obj: Optional[Union[str, "os.PathLike[str]", IO[bytes]]] = None,
        stamp: str = "CCHDHYDRO",
        use_source_c_format: bool = True,
    ):
        """Export :class:`hydro.exchange.Exchange` object to WHP-Exchange datafile(s)"""
        log.info(f"Converting {self} to exchange csv")
        log.info(f"File Type is {self.file_type.name}")

        if self.file_type == FileType.CTD and len(self) > 1:
            if filename_or_obj is None:
                raise ValueError(
                    "Will result in multiple files, please provide a path or open filelike object in bytes mode to write to"
                )
            with ZipFile(filename_or_obj, "w", compression=ZIP_DEFLATED) as zf:
                for profile in self.iter_profiles():
                    with zf.open(profile.gen_fname(), "w") as zip_prof:
                        zip_prof.write(profile.to_exchange_csv())
            return

        # File Fortmat indicator
        # https://exchange-format.readthedocs.io/en/latest/common.html#file-format-indicator
        now = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        file_format_indicator = f"{self.file_type.name},{now}{stamp}"

        # File Comments
        # https://exchange-format.readthedocs.io/en/latest/common.html#optional-comment-lines
        file_comments = "\n".join([f"#{line}" for line in self.comments.splitlines()])

        # File Parameter and Unit Lines
        # https://exchange-format.readthedocs.io/en/latest/common.html#parameter-and-unit-lines
        _parameters = []
        _units = []

        def none_to_empty(x: Optional[str]) -> str:
            if x is None:
                return ""
            return x

        _data_params = self.parameters
        _ctd_params: Tuple[WHPName, ...] = tuple()

        if self.file_type == FileType.CTD:
            _ctd_params = tuple(filter(lambda x: x.scope == "profile", _data_params))
            _data_params = tuple(filter(lambda x: x.scope != "profile", _data_params))

        for param in _data_params:
            _parameters.append(param.whp_name)
            _units.append(none_to_empty(param.whp_unit))

            if param in self.flags:
                _parameters.append(f"{param.whp_name}_FLAG_W")
                _units.append("")

            if param in self.errors:
                if param.error_name is None:
                    raise ValueError(f"No error name found for param {param}")
                _parameters.append(param.error_name)
                _units.append(none_to_empty(param.whp_unit))

        file_parameters = ",".join(_parameters)
        file_units = ",".join(_units)

        _ctd_headers = []
        if self.file_type == FileType.CTD:
            _ctd_headers.append(f"NUMBER_HEADERS = {len(_ctd_params) + 1}")
            for param in _ctd_params:
                ctd_source_c_format: Optional[int] = None
                if use_source_c_format:
                    ctd_source_c_format = self._param_source_numeric_precision.get(
                        param
                    )

                ctdh_key = param.whp_name
                ctdh_value = self.parameter_to_ndarray(param)[
                    self.ndaray_indicies[self.keys[0]]
                ]
                _ctd_headers.append(
                    f"{ctdh_key} = {param.strfex(ctdh_value, numeric_precision_override=ctd_source_c_format).strip()}"
                )

        file_ctd_headers = "\n".join(_ctd_headers)

        _data = []
        for key in self.keys:
            row = []
            for param in _data_params:

                source_c_format: Optional[int] = None
                if use_source_c_format:
                    source_c_format = self._param_source_numeric_precision.get(param)

                value = self.parameter_to_ndarray(param)[self.ndaray_indicies[key]]
                row.append(
                    param.strfex(value, numeric_precision_override=source_c_format)
                )

                if param in self.flags:
                    value = self.flag_to_ndarray(param)[self.ndaray_indicies[key]]
                    row.append(param.strfex(value, flag=True))

                if param in self.errors:
                    value = self.error_to_ndarray(param)[self.ndaray_indicies[key]]
                    row.append(
                        param.strfex(value, numeric_precision_override=source_c_format)
                    )
            _data.append(",".join(row))
        file_data = "\n".join(_data)

        _data_file = [
            file_format_indicator,
            file_comments,
            file_parameters,
            file_units,
            file_data,
            "END_DATA",
        ]
        if self.file_type == FileType.CTD:
            _data_file.insert(2, file_ctd_headers)

        final_file = "\n".join(_data_file).encode("utf8")

        if filename_or_obj is None:
            return final_file

        # we are going to ignore typing here since the open call will use
        # the first two types: str, and PathLike. If it is an open file-like
        # the second write will work unless the wrong modes are set (not binary
        # or writable), just raise if things are wrong.
        try:
            with open(filename_or_obj, "wb") as f:  # type: ignore
                f.write(final_file)
        except TypeError:
            filename_or_obj.write(final_file)  # type: ignore
