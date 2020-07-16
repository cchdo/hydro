from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import (
    Union,
    Tuple,
    Optional,
    Dict,
    NamedTuple,
)
from datetime import datetime
from enum import Enum, auto
from itertools import groupby
from functools import cached_property

import numpy as np
import pandas as pd
import xarray as xr

from hydro.data import WHPNames, WHPName
from .flags import ExchangeBottleFlag, ExchangeSampleFlag, ExchangeCTDFlag
from .exceptions import (
    ExchangeDataFlagPairError,
    ExchangeDataPartialCoordinateError,
    ExchangeDataInconsistentCoordinateError,
)

try:
    from hydro import __version__ as hydro_version
except ImportError:
    hydro_version = "unknown"


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
            flag = FLAG_SCHEME[whpname.flag_w](flag_v)  # type: ignore
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

        date = datetime.strptime(data_line.pop(cls.DATE).data, "%Y%m%d").date()
        try:
            time = data_line.pop(cls.TIME).data
            time_obj = datetime.strptime(time, "%H%M").time()
            date = datetime.combine(date, time_obj)
        except KeyError:
            pass

        return cls(
            x=ExchangeDataPoint.from_ir(cls.LONGITUDE, data_line.pop(cls.LONGITUDE)),
            y=ExchangeDataPoint.from_ir(cls.LATITUDE, data_line.pop(cls.LATITUDE)),
            z=ExchangeDataPoint.from_ir(cls.CTDPRS, data_line[cls.CTDPRS]),
            t=np.datetime64(date),
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

    _ndarray_cache: Dict[Tuple[WHPName, str], np.ndarray] = field(
        default_factory=dict, init=False
    )

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

        attrs = {
            "standard_name": "status_flag",
            "flag_values": np.array(flag_values, dtype="int8"),
            "flag_meanings": " ".join(flag_meanings),
            "whp_flag_scheme": param.flag_w,
        }

        da = xr.DataArray(data=data, dims=dims, attrs=attrs, name=name)

        da.encoding["dtype"] = "int8"
        da.encoding["_FillValue"] = 9

        return da

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
        da.encoding["dtype"] = "float32"

        return da

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

    def time_to_dataarray(self) -> xr.DataArray:
        data = self.time_to_ndarray()[:, 0]
        # units will be handeled by xarray on serialization
        attrs = {"standard_name": "time", "axis": "T", "whp_name": ["DATE", "TIME"]}
        dims = DIMS[: data.ndim]
        da = xr.DataArray(name="time", data=data, dims=dims, attrs=attrs,)
        da.encoding["_FillValue"] = None
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

        axis = ExchangeXYZT.CF_AXIS[param]
        attrs["axis"] = axis
        if axis == "Z":
            attrs["positive"] = "down"

        dims = DIMS[: data.ndim]
        name = axis_to_name[axis]

        da = xr.DataArray(name=name, data=data, dims=dims, attrs=attrs,)
        if not np.any(np.isnan(data)):
            da.encoding["_FillValue"] = None

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

        da = xr.DataArray(name=name, data=data, dims=dims, attrs=attrs,)

        if param.data_type == int:  # type: ignore
            # the woce spec says this should go from 1 and incriment
            # largest I have seen is maybe 20something on a GT cruise
            da.encoding["dtype"] = "int8"

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

    def parameter_to_ndarray(self, param: WHPName) -> np.ndarray:
        try:
            return self._ndarray_cache[(param, "value")]
        except KeyError:
            pass

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

        self._ndarray_cache[(param, "value")] = arr
        return arr

    def parameter_to_dataarray(
        self, param: WHPName, name: Optional[str] = None
    ) -> xr.DataArray:
        data = self.parameter_to_ndarray(param)

        if param.scope == "profile":
            data = data[:, 0]

        dims = DIMS[: data.ndim]

        attrs = param.get_nc_attrs()

        da = xr.DataArray(data=data, dims=dims, attrs=attrs, name=name)

        if data.dtype == object:
            da.encoding["dtype"] = "str"
        if data.dtype == float:
            da.encoding["dtype"] = "float32"

        return da

    def iter_profile_coordinates(self):
        for profile in self.iter_profiles():
            yield profile.coordinates[profile.keys[-1]]

    def to_xarray(self):
        """
        A lingering special case...
        If present, bottle trip information:

        * BTL_LAT
        * BTL_LON
        * BTL_DATE
        * BTL_TIME

        Note that the seperate date and time need to be combined into a single
        date var for CF. Except for the bottle trip information, all the
        above should probably get "real" var names not just var0, ..., varN.
        """

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

        # CCHDO Sample Identifying parameters go into coords
        for param in ExchangeCompositeKey.WHP_PARAMS:
            consumed.append(param)
            coord = self.key_to_dataarray(param)
            coords[coord.name] = coord

        data_params = (param for param in self.parameters if param not in consumed)
        for n, param in enumerate(data_params):
            da = self.parameter_to_dataarray(param, name=f"var{n}")
            data_arrays.append(da)

            ancillary_variables = []

            if param in self.flags:
                da_qc = self.flag_to_dataarray(param, name=f"var{n}_qc")
                data_arrays.append(da_qc)

                ancillary_variables.append(da_qc.name)

            if param in self.errors:
                da_error = self.error_to_dataarray(param, name=f"var{n}_error")
                data_arrays.append(da_error)

                ancillary_variables.append(da_error.name)

            if len(ancillary_variables) > 0:
                da.attrs["ancillary_variables"] = " ".join(ancillary_variables)

        dataset = xr.Dataset(
            {da.name: da for da in data_arrays},
            coords=coords,
            attrs={"Conventions": f"CF-1.8 CCHDO-{hydro_version}"},
        )
        return dataset
