from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import (
    Union,
    Tuple,
    Optional,
    Dict,
    NamedTuple,
    Literal,
)
from datetime import datetime
from enum import Enum, auto
from operator import attrgetter
from itertools import groupby
from functools import cached_property

import numpy as np
import pandas as pd

from hydro.data import WHPNames, WHPName
from .flags import ExchangeBottleFlag, ExchangeSampleFlag, ExchangeCTDFlag
from .exceptions import (
    ExchangeDataFlagPairError,
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


class ExchangeDataProxy(Mapping):
    def __init__(
        self, exchange: Exchange, attr: Literal["value", "flag", "error"] = "value"
    ):
        self._ex = exchange
        self._get = attrgetter(attr)

    def __getitem__(self, key: Tuple[ExchangeCompositeKey, WHPName]):
        row, col = key
        if col in ExchangeCompositeKey.WHP_PARAMS:
            return row[col]
        elif col in ExchangeXYZT.WHP_PARAMS:
            return self._ex.coordinates[row][col]
        else:
            try:
                return self._get(self._ex.data[row][col])
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
        return self.shape[0]

    @cached_property
    def at(self) -> ExchangeDataProxy:
        return ExchangeDataProxy(self)

    @cached_property
    def at_flag(self) -> ExchangeDataProxy:
        return ExchangeDataProxy(self, "flag")

    @cached_property
    def at_error(self) -> ExchangeDataProxy:
        return ExchangeDataProxy(self, "error")

    @cached_property
    def shape(self):
        x = len({key.profile_id for key in self.keys})
        y = max([len(prof.keys) for prof in self.iter_profiles()])
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

        for row, (_key, group) in enumerate(groupby(self.keys, lambda k: k.profile_id)):
            for col, key in enumerate(group):
                arr[row, col] = self.at_flag[(key, param)]

        if arr.shape[0] == 1:
            return np.squeeze(arr)
        return arr

    def error_to_ndarray(self, param: WHPName) -> np.ndarray:
        if param not in self.errors:
            raise KeyError(f"No error for {param}")

        arr = np.full(self.shape, np.nan, dtype=float)

        for row, (_key, group) in enumerate(groupby(self.keys, lambda k: k.profile_id)):
            for col, key in enumerate(group):
                arr[row, col] = self.at_error[(key, param)]

        if arr.shape[0] == 1:
            return np.squeeze(arr)
        return arr

    def time_to_ndarray(self) -> np.ndarray:
        """Time is a specal/funky case

        .. todo::

            Write why time is specal in exchange
        """
        arr = np.full(self.shape, np.datetime64("NaT"), dtype="datetime64[m]")

        for row, (_key, group) in enumerate(groupby(self.keys, lambda k: k.profile_id)):
            for col, key in enumerate(group):
                arr[row, col] = self.coordinates[key].t

        if arr.shape[0] == 1:
            return np.squeeze(arr)
        return arr

    def parameter_to_ndarray(self, param: WHPName) -> np.ndarray:
        # https://github.com/python/mypy/issues/5485
        dtype = param.data_type  # type: ignore
        if dtype == str:
            arr = np.full(self.shape, "", dtype=object)
        else:
            arr = np.full(self.shape, np.nan, dtype=float)
        for row, (_key, group) in enumerate(groupby(self.keys, lambda k: k.profile_id)):
            for col, key in enumerate(group):
                arr[row, col] = self.at[(key, param)]
        if arr.shape[0] == 1:
            return np.squeeze(arr)
        return arr

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

        consumed = []
        data_arrays = []

        # specal cases
        # for key in ExchangeXYZT.WHP_PARAMS:
        for param in self.parameters:
            if param in ExchangeXYZT.WHP_PARAMS:
                consumed.append(param)
        coords = {}
        coords["latitude"] = xr.DataArray(
            self.parameter_to_ndarray(ExchangeXYZT.LATITUDE)[:, 0],
            dims=("N_PROF",),
            attrs={
                "standard_name": ExchangeXYZT.LATITUDE.cf_name,
                "axis": "Y",
                "whp_name": ExchangeXYZT.LATITUDE.whp_name,
            },
        )
        coords["longitude"] = xr.DataArray(
            self.parameter_to_ndarray(ExchangeXYZT.LONGITUDE)[:, 0],
            dims=("N_PROF",),
            attrs={
                "standard_name": ExchangeXYZT.LONGITUDE.cf_name,
                "axis": "X",
                "whp_name": ExchangeXYZT.LONGITUDE.whp_name,
            },
        )
        coords["time"] = xr.DataArray(
            self.time_to_ndarray()[:, 0],
            dims=("N_PROF",),
            attrs={"standard_name": "time", "axis": "T", "whp_name": ["DATE", "TIME"]},
        )
        coords["pressure"] = xr.DataArray(
            self.parameter_to_ndarray(ExchangeXYZT.CTDPRS),
            dims=("N_PROF", "N_LEVELS"),
            attrs={
                "standard_name": ExchangeXYZT.CTDPRS.cf_name,
                "axis": "Z",
                "positive": "down",
                "whp_name": ExchangeXYZT.CTDPRS.whp_name,
                "whp_unit": ExchangeXYZT.CTDPRS.whp_unit,
            },
        )

        for coord in coords.values():
            coord.encoding["_FillValue"] = None

        data_params = (param for param in self.parameters if param not in consumed)
        for n, param in enumerate(data_params):
            if param.scope == "profile":
                values = self.parameter_to_ndarray(param)[:, 0]
                dims = ("N_PROF",)
            else:
                values = self.parameter_to_ndarray(param)
                dims = ("N_PROF", "N_LEVELS")

            attrs = {}
            if param.cf_name is not None:
                attrs["standard_name"] = param.cf_name

            data_array = xr.DataArray(values, dims=dims, name=f"var{n}", attrs=attrs)

            if data_array.dtype == object:
                data_array.encoding["dtype"] = "str"
            if data_array.dtype == float:
                data_array.encoding["dtype"] = "float32"

            data_arrays.append(data_array)

            if param in self.flags:
                values = self.flag_to_ndarray(param)
                dims = ("N_PROF", "N_LEVELS")
                anc_data_array = xr.DataArray(values, dims=dims, name=f"var{n}_qc")
                anc_data_array.attrs["standard_name"] = "status_flag"
                anc_data_array.encoding["dtype"] = "int8"
                anc_data_array.encoding["_FillValue"] = 9
                data_arrays.append(anc_data_array)

                data_array.attrs["ancillary_variables"] = anc_data_array.name

        dataset = xr.Dataset({da.name: da for da in data_arrays}, coords=coords)
        return dataset
