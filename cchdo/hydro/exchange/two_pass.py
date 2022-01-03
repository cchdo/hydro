from collections import defaultdict
import logging
import io
import dataclasses
from collections.abc import Mapping
from typing import Tuple, Dict
from operator import attrgetter
from functools import cached_property
from itertools import chain
from zipfile import ZipFile, is_zipfile


import numpy as np
import numpy.typing as npt
import pandas as pd

import xarray as xr

from cchdo.params import WHPName, WHPNames

from .exceptions import (
    ExchangeEncodingError,
    ExchangeBOMError,
    ExchangeLEError,
    ExchangeMagicNumberError,
    ExchangeDuplicateParameterError,
    ExchangeParameterUnitAlignmentError,
    ExchangeDataPartialCoordinateError,
)
from .containers import FileType, ExchangeCompositeKey

from .io import (
    _bottle_get_params,
    _bottle_get_flags,
    _bottle_get_errors,
    _ctd_get_header,
)

log = logging.getLogger(__name__)

EXPOCODE = WHPNames["EXPOCODE"]
STNNBR = WHPNames["STNNBR"]
CASTNO = WHPNames["CASTNO"]

## tmp for new implimentatio
@dataclasses.dataclass(frozen=True)
class ExchangeXYZT(Mapping):
    x: float  # Longitude
    y: float  # Latitude
    z: float  # Pressure
    t: np.datetime64  # Time obviously...

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

    def __repr__(self):

        return (
            f"ExchangeXYZT("
            f"x={self.x} "
            f"y={self.y} "
            f"z={self.z} "
            f"t={self.t!r})"
        )

    def __post_init__(self):
        if not all(
            [
                self.x is not None,
                self.y is not None,
                self.z is not None,
            ]
        ):
            raise ExchangeDataPartialCoordinateError(self)

    @property
    def _mapping(self):
        return (
            {
                self.LONGITUDE: self.x,
                self.LATITUDE: self.y,
                self.CTDPRS: self.z,
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
        return (self.t, self.z, self.x, self.y) < (
            other.t,
            other.z,
            other.x,
            other.y,
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


@dataclasses.dataclass
class ExchangeData:
    single_profile: bool
    param_cols: Dict[WHPName, np.ndarray]
    flag_cols: Dict[WHPName, np.ndarray]
    error_cols: Dict[WHPName, np.ndarray]

    # OG Print Precition Tracking
    param_precisions: Dict[WHPName, int]
    error_precisions: Dict[WHPName, int]

    comments: str

    def __post_init__(self):
        # check the shapes of all the nd arrays are the same
        get_shape = attrgetter("shape")
        shapes = [
            get_shape(arr)
            for arr in chain(
                self.param_cols.values(),
                self.flag_cols.values(),
                self.error_cols.values(),
            )
        ]
        if not all([shape == shapes[0] for shape in shapes]):
            # TODO Error handling
            raise ValueError("shape error")

        self.shape = shapes[0]

        if self.single_profile:
            # all "profile scoped" params must have the same values
            for param, data in self.param_cols.items():
                if param.scope != "profile":
                    continue
                if not np.unique(data).shape[0] == 1:
                    raise ValueError("inconsistent param")

        # make sure flags and errors are strict subsets
        if not self.flag_cols.keys() <= self.param_cols.keys():
            raise ValueError("orphan flag")
        if not self.error_cols.keys() <= self.param_cols.keys():
            raise ValueError("orphan error")

    def split_profiles(self):
        expocode = self.param_cols[EXPOCODE]
        station = self.param_cols[STNNBR]
        cast = self.param_cols[CASTNO]

        # need to split up by profiles and _not_ assume the bottles are in order
        # use the actual values to sort things out
        # we don't care what the values are, they just need to work
        log.debug("Grouping Profiles by Key")
        prof_ids = np.char.add(np.char.add(expocode, station), cast.astype("U"))
        unique_profile_ids = np.unique(prof_ids)
        log.debug("Found %s unique profile keys", len(unique_profile_ids))
        profiles = [np.nonzero(prof_ids == prof) for prof in unique_profile_ids]

        log.debug("Actually splitting profiles")
        return [
            ExchangeData(
                single_profile=True,
                param_cols={
                    param: data[profile] for param, data in self.param_cols.items()
                },
                flag_cols={
                    param: data[profile] for param, data in self.flag_cols.items()
                },
                error_cols={
                    param: data[profile] for param, data in self.error_cols.items()
                },
                param_precisions=self.param_precisions,
                error_precisions=self.error_precisions,
                comments=self.comments,
            )
            for profile in profiles
        ]

    @property
    def parameters(self):
        return self.param_cols.keys()

    @cached_property
    def str_lens(self) -> Dict[WHPName, int]:
        np_char_size = np.dtype("U1").itemsize
        lens = {}
        for param, data in self.param_cols.items():
            if param.dtype == "string":
                lens[param] = data.itemsize // np_char_size

        return lens


@dataclasses.dataclass
class ExchangeInfo:
    stamp_line: int
    comments_start: int
    comments_end: int
    ctd_header_start: int
    ctd_header_end: int
    params_line: int
    units_line: int
    data_start: int
    data_end: int
    post_data_start: int
    post_data_end: int
    _raw_lines: Tuple[str, ...] = dataclasses.field(repr=False)

    @property
    def comments(self):
        return slice(self.comments_start, self.comments_end)

    @property
    def data(self):
        return slice(self.data_start, self.data_end)

    @property
    def post_data(self):
        return slice(self.post_data_start, self.post_data_end)

    @property
    def ctd_headers(self):
        slc = slice(self.ctd_header_start, self.ctd_header_end)
        return dict([_ctd_get_header(line) for line in self._raw_lines[slc]])

    @cached_property
    def params(self):
        ctd_params = self.ctd_headers.keys()
        return [
            param.strip()
            for param in [*ctd_params, *self._raw_lines[self.params_line].split(",")]
        ]

    @cached_property
    def units(self):
        # we can have a bunch of empty strings as units, we want these to be
        # None to match what would be in a WHPName object
        ctd_units = [None for _ in self.ctd_headers]
        return [
            x if x != "" else None
            for x in [
                *ctd_units,
                *[unit.strip() for unit in self._raw_lines[self.units_line].split(",")],
            ]
        ]

    @cached_property
    def whp_params(self):
        # TODO remove when min pyver is 3.10
        if len(self.params) != len(set(self.params)):
            raise ExchangeDuplicateParameterError

        # In initial testing, it was discovered that approx half the ctd files
        # had trailing commas in just the params and units lines
        if self.params[-1] == "" and self.units[-1] is None:
            log.warning(
                "Removed trailing empty param/unit pair, this indicates these lines have trailing commas."
            )
            self.params.pop()
            self.units.pop()

        # the number of expected columns is just going to be the number of
        # parameter names we see
        column_count = len(self.params)

        if len(self.units) != column_count:
            raise ExchangeParameterUnitAlignmentError

        return _bottle_get_params(zip(self.params, self.units))

    @cached_property
    def whp_flags(self):
        self.whp_params
        return _bottle_get_flags(zip(self.params, self.units), self.whp_params)

    @cached_property
    def whp_errors(self):
        self.whp_params
        return _bottle_get_errors(zip(self.params, self.units), self.whp_params)

    # TODO cache?
    @property
    def _np_data_block(self):
        _raw_data = tuple(
            tuple((*self.ctd_headers.values(), *line.replace(" ", "").split(",")))
            for line in self._raw_lines[self.data]
        )
        return np.array(_raw_data, dtype="U")

    def finalize(self):
        # TODO clean up this function
        log.debug("Finializing...")
        single_profile = any(self.ctd_headers)

        np_db = self._np_data_block

        dtype_map = {"string": "U", "integer": "float32", "decimal": "float64"}

        whp_param_cols = {}
        whp_flag_cols = {}
        whp_error_cols = {}
        whp_param_precisions = {}
        whp_error_precisions = {}

        for param, idx in self.whp_params.items():
            param_col = np_db[:, idx]
            if param.dtype == "decimal":
                whp_param_precisions[param] = _extract_numeric_precisions(param_col)
                fill_spaces = np.char.startswith(param_col, "-999")
                param_col[fill_spaces] = "nan"
            whp_param_cols[param] = param_col.astype(dtype_map[param.dtype])
        for param, idx in self.whp_flags.items():
            fill_spaces = np.char.startswith(param_col, "9")
            param_col[fill_spaces] = "nan"
            whp_flag_cols[param] = np_db[:, idx].astype("float16")
        for param, idx in self.whp_errors.items():
            param_col = np_db[:, idx]
            if param.dtype == "decimal":
                whp_error_precisions[param] = _extract_numeric_precisions(param_col)
                fill_spaces = np.char.startswith(param_col, "-999")
                param_col[fill_spaces] = "nan"
            whp_error_cols[param] = param_col.astype(dtype_map[param.dtype])
        del self._raw_lines

        return ExchangeData(
            single_profile,
            whp_param_cols,
            whp_flag_cols,
            whp_error_cols,
            whp_param_precisions,
            whp_error_precisions,
            comments="test",
        )


def _get_parts(lines: Tuple[str, ...], ftype: FileType) -> ExchangeInfo:
    """Figure out the line numbers/indicies of the parts of the exchange file"""
    stamp = 0  # file stamp is always the first line of a valid exchange
    comments_start = 1
    comments_end = 1
    ctd_header_start = 1
    ctd_header_end = 1
    params = 1
    units = 1
    data_start = 1
    data_end = 1
    post_data_start = 1
    post_data_end = 1

    looking_for = "file_stamp"
    ctd_num_headers = 0

    log.debug("Looking for file parts")

    for idx, line in enumerate(lines):
        if looking_for == "file_stamp":
            looking_for = "comments"
            continue

        if looking_for == "comments":
            if line.startswith("#"):
                comments_end = idx + 1
            elif ftype == FileType.CTD:
                looking_for = "ctd_headers"
                param, value = _ctd_get_header(line, dtype=int)
                if param != "NUMBER_HEADERS":
                    raise ValueError()
                ctd_num_headers = value - 1
                ctd_header_start = idx + 1
                continue
            else:
                looking_for = "params"
                continue

        if looking_for == "ctd_headers":
            if ctd_num_headers == 0:
                ctd_header_end = idx
                looking_for = "params"
                continue
            ctd_num_headers -= 1

        if looking_for == "params":
            params = idx - 1
            looking_for = "units"
            continue

        if looking_for == "units":
            units = idx - 1
            data_start = idx
            looking_for = "data"
            continue

        if looking_for == "data":
            if line == "END_DATA":
                data_end = idx

                looking_for = "post_data"
                post_data_start = post_data_end = idx + 1
                continue

        if looking_for == "post_data":
            post_data_end = idx

    return ExchangeInfo(
        stamp_line=stamp,
        comments_start=comments_start,
        comments_end=comments_end,
        ctd_header_start=ctd_header_start,
        ctd_header_end=ctd_header_end,
        params_line=params,
        units_line=units,
        data_start=data_start,
        data_end=data_end,
        post_data_start=post_data_start,
        post_data_end=post_data_end,
        _raw_lines=lines,
    )


def _extract_numeric_precisions(data: npt.ArrayLike) -> np.ndarray:
    """Get the numeric precision of a printed decimal number"""
    # magic number explain: np.char.partition expands each element into a 3-tuple
    # of (pre, sep, post) of some sep, in our case a "." char.
    # We only want the post bits [idx 2] (the number of chars after a decimal seperator)
    # of the last axis.
    numeric_parts = np.char.partition(data, ".")[..., 2]
    str_lens = np.char.str_len(numeric_parts)
    return np.max(str_lens, axis=0)


def _get_ex_keys(data: Dict[WHPName, np.ndarray]):

    # get the col indicies
    # TODO make this a classmethod of ExchangeCompositeKey?
    key_cols = zip(
        data[ExchangeCompositeKey.EXPOCODE],
        data[ExchangeCompositeKey.STNNBR],
        data[ExchangeCompositeKey.CASTNO],
        data[ExchangeCompositeKey.SAMPNO],
    )

    exchange_keys = []
    for idx, (expocode, station, cast, sample) in enumerate(key_cols):
        exchange_keys.append(
            (
                ExchangeCompositeKey(
                    expocode=ExchangeCompositeKey.EXPOCODE.data_type(expocode),
                    station=ExchangeCompositeKey.STNNBR.data_type(station),
                    cast=ExchangeCompositeKey.CASTNO.data_type(cast),
                    sample=ExchangeCompositeKey.SAMPNO.data_type(sample),
                ),
                idx,
            )
        )

    return exchange_keys


def _get_ex_xyzt(data: Dict[WHPName, np.ndarray]):
    # fixup date/time to deal with the ability to _not_ have time just a date
    if ExchangeXYZT.TIME not in data:
        time = pd.to_datetime(data[ExchangeXYZT.DATE], format="%Y%m%d").values.astype(
            "datetime64[D]"
        )
    else:
        cat_time = np.char.add(data[ExchangeXYZT.DATE], data[ExchangeXYZT.TIME])
        time = pd.to_datetime(cat_time, format="%Y%m%d%H%M").values.astype(
            "datetime64[m]"
        )

    cols = zip(
        data[ExchangeXYZT.LONGITUDE],
        data[ExchangeXYZT.LATITUDE],
        data[ExchangeXYZT.CTDPRS],
        time,
    )

    ex_xyzt = []
    for idx, (lon, lat, pres, time) in enumerate(cols):
        ex_xyzt.append((ExchangeXYZT(lon, lat, pres, time), idx))

    return ex_xyzt


def read_exchange(path):

    log.info("Trying to open as local filepath")
    with open(path, "rb") as f:
        data_raw = io.BytesIO(f.read())

    data = []
    if is_zipfile(data_raw):

        data_raw.seek(0)  # is_zipfile moves the "tell" position
        with ZipFile(data_raw) as zf:
            for zipinfo in zf.infolist():
                log.debug("Reading %s", zipinfo)
                try:
                    data.append(zf.read(zipinfo).decode("utf8"))
                except UnicodeDecodeError as error:
                    raise ExchangeEncodingError from error
    else:
        data_raw.seek(0)  # is_zipfile moves the "tell" position
        try:
            data.append(data_raw.read().decode("utf8"))
        except UnicodeDecodeError as error:
            raise ExchangeEncodingError from error

    # cleanup the data_raw to free the memory
    data_raw.close()

    log.info("Checking for BOM")
    if any((df.startswith("\ufeff") for df in data)):
        raise ExchangeBOMError

    log.info("Checking Line Endings")
    if any(("\r" in df for df in data)):
        raise ExchangeLEError

    log.info("Detecting file type")
    if all((df.startswith("BOTTLE") for df in data)):
        ftype = FileType.BOTTLE
    elif all((df.startswith("CTD") for df in data)):
        ftype = FileType.CTD
    else:
        # TODO this is where the "mixed" check is happening now
        raise ExchangeMagicNumberError

    log.info("Found filetype: %s", ftype.name)

    exchange_data = [
        _get_parts(tuple(df.splitlines()), ftype=ftype).finalize() for df in data
    ]

    if not all((fp.single_profile for fp in exchange_data)):
        exchange_data = list(chain(*[exd.split_profiles() for exd in exchange_data]))

    N_PROF = len(exchange_data)
    N_LEVELS = max((fp.shape[0] for fp in exchange_data))

    log.debug((N_PROF, N_LEVELS))

    # TODO sort profiles

    params = set(chain(*[exd.param_cols.keys() for exd in exchange_data]))
    flags = set(chain(*[exd.flag_cols.keys() for exd in exchange_data]))
    errors = set(chain(*[exd.error_cols.keys() for exd in exchange_data]))
    log.debug("Dealing with strings")
    str_len = 1
    for exd in exchange_data:
        for param, value in exd.str_lens.items():
            str_len = max(value, str_len)

    dataarrays = {}
    dtype_map = {"string": f"U{str_len}", "integer": "float32", "decimal": "float64"}
    fills_map = {"string": "", "integer": np.nan, "decimal": np.nan}

    log.debug("Init DataArrays")
    for param in params:
        if param.scope == "profile":
            arr = np.full(
                (N_PROF),
                fill_value=fills_map[param.dtype],
                dtype=dtype_map[param.dtype],
            )
        elif param.scope == "sample":
            arr = np.full(
                (N_PROF, N_LEVELS),
                fill_value=fills_map[param.dtype],
                dtype=dtype_map[param.dtype],
            )
        dataarrays[param.nc_name] = xr.DataArray(arr, attrs=param.get_nc_attrs())

        if param in flags:
            if param.scope == "profile":
                arr = np.full((N_PROF), fill_value=np.nan, dtype="float32")
            elif param.scope == "sample":
                arr = np.full((N_PROF, N_LEVELS), fill_value=np.nan, dtype="float32")
            dataarrays[f"{param.nc_name}_qc"] = xr.DataArray(
                arr, attrs=param.get_nc_attrs(error=True)
            )

    for n_prof, exd in enumerate(exchange_data):
        for param in params:
            if param.scope == "profile":
                dataarrays[param.nc_name][n_prof] = exd.param_cols[param][0]
            if param.scope == "sample":
                data = exd.param_cols[param]
                dataarrays[param.nc_name][n_prof, : len(data)] = data
            if param in flags:
                if param.scope == "profile":
                    dataarrays[f"{param.nc_name}_qc"][n_prof] = exd.flag_cols[param][0]
                if param.scope == "sample":
                    data = exd.flag_cols[param]
                    dataarrays[f"{param.nc_name}_qc"][n_prof, : len(data)] = data

    return xr.Dataset(dataarrays)
