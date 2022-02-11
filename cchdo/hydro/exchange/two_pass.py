import logging
import io
import dataclasses
from typing import Tuple, Dict, Union, Optional
from operator import attrgetter
from functools import cached_property
from itertools import chain
from zipfile import ZipFile, is_zipfile
from pathlib import Path
from datetime import datetime
from enum import Enum, auto

import requests
import numpy as np
import numpy.typing as npt

import xarray as xr

from cchdo.params import WHPName, WHPNames
from cchdo.params._version import version as params_version

from .exceptions import (
    ExchangeDataInconsistentCoordinateError,
    ExchangeEncodingError,
    ExchangeBOMError,
    ExchangeLEError,
    ExchangeMagicNumberError,
    ExchangeDuplicateParameterError,
    ExchangeParameterUnitAlignmentError,
)
from .containers import FileType
from .flags import ExchangeBottleFlag, ExchangeCTDFlag, ExchangeSampleFlag

from .io import (
    _bottle_get_params,
    _bottle_get_flags,
    _bottle_get_errors,
    _ctd_get_header,
)

try:
    from .. import __version__ as hydro_version

    CCHDO_VERSION = ".".join(hydro_version.split(".")[:2])
    if "dev" in hydro_version:
        CCHDO_VERSION = hydro_version
except ImportError:
    hydro_version = CCHDO_VERSION = "unknown"

log = logging.getLogger(__name__)

DIMS = ("N_PROF", "N_LEVELS")

EXPOCODE = WHPNames["EXPOCODE"]
STNNBR = WHPNames["STNNBR"]
CASTNO = WHPNames["CASTNO"]
SAMPNO = WHPNames["SAMPNO"]
DATE = WHPNames["DATE"]
TIME = WHPNames["TIME"]
LATITUDE = WHPNames["LATITUDE"]
LONGITUDE = WHPNames["LONGITUDE"]
CTDPRS = WHPNames[("CTDPRS", "DBAR")]

COORDS = [
    EXPOCODE,
    STNNBR,
    CASTNO,
    SAMPNO,
    DATE,
    TIME,
    LATITUDE,
    LONGITUDE,
    CTDPRS,
]

FLAG_SCHEME = {
    "woce_bottle": ExchangeBottleFlag,
    "woce_discrete": ExchangeSampleFlag,
    "woce_ctd": ExchangeCTDFlag,
}

GEOMETRY_VARS = ("expocode", "station", "cast", "section_id", "time")


def add_geometry_var(dataset: xr.Dataset) -> xr.Dataset:
    """Adds a CF-1.8 Geometry container variable to the dataset

    This allows for compatabiltiy with tools like gdal
    """
    geometry_var = xr.DataArray(
        name="geometry_container",
        attrs={
            "geometry_type": "point",
            "node_coordinates": "longitude latitude",
        },
    )
    dataset["geometry_container"] = geometry_var

    for var in GEOMETRY_VARS:
        if var in dataset:
            dataset[var].attrs["geometry"] = "geometry_container"

    return dataset


def add_profile_type(dataset: xr.Dataset, ftype: FileType) -> xr.Dataset:
    """Adds a `profile_type` string variable to the dataset.

    This is for ODV compatability

    .. warning::
      Currently mixed profile types are not supported
    """
    profile_type = xr.DataArray(
        np.full(dataset.dims["N_PROF"], fill_value=ftype.value, dtype="U1"),
        name="profile_type",
        dims=DIMS[0],
    )
    profile_type.encoding["dtype"] = "S1"

    dataset["profile_type"] = profile_type
    return dataset


@dataclasses.dataclass
class ExchangeData:
    single_profile: bool
    param_cols: Dict[WHPName, np.ndarray]
    flag_cols: Dict[WHPName, np.ndarray]
    error_cols: Dict[WHPName, np.ndarray]

    # OG Print Precition Tracking
    param_precisions: Dict[WHPName, npt.NDArray[np.int_]]
    error_precisions: Dict[WHPName, npt.NDArray[np.int_]]

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
    """Low level dataclass containing the parts of an exchange file"""

    stamp_slice: slice
    comments_slice: slice
    ctd_headers_slice: slice
    params_idx: int
    units_idx: int
    data_slice: slice
    post_data_slice: slice
    _raw_lines: Tuple[str, ...] = dataclasses.field(repr=False)

    @property
    def stamp(self):
        return self._raw_lines[self.stamp_slice]

    @property
    def comments(self):
        raw_comments = self._raw_lines[self.comments_slice]
        return [c[1:] if c.startswith("#") else c for c in raw_comments]

    @property
    def ctd_headers(self):
        return dict(
            [_ctd_get_header(line) for line in self._raw_lines[self.ctd_headers_slice]]
        )

    @cached_property
    def params(self):
        ctd_params = self.ctd_headers.keys()
        data_params = self._raw_lines[self.params_idx].split(",")
        return [param.strip() for param in [*ctd_params, *data_params]]

    @cached_property
    def units(self):
        # we can have a bunch of empty strings as units, we want these to be
        # None to match what would be in a WHPName object
        ctd_units = [None for _ in self.ctd_headers]
        data_units = self._raw_lines[self.units_idx].split(",")
        return [
            x if x != "" else None
            for x in [
                *ctd_units,
                *[unit.strip() for unit in data_units],
            ]
        ]

    @property
    def data(self):
        return self._raw_lines[self.data_slice]

    @property
    def post_data(self):
        return self._raw_lines[self.post_data_slice]

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
        return _bottle_get_flags(zip(self.params, self.units), self.whp_params)

    @cached_property
    def whp_errors(self):
        return _bottle_get_errors(zip(self.params, self.units), self.whp_params)

    @property
    def _np_data_block(self):
        _raw_data = tuple(
            tuple((*self.ctd_headers.values(), *line.replace(" ", "").split(",")))
            for line in self.data
        )
        return np.array(_raw_data, dtype="U")

    def finalize(self) -> ExchangeData:
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
            fill_spaces = np.char.startswith(param_col, "-999")
            if param.dtype == "decimal":
                whp_param_precisions[param] = _extract_numeric_precisions(param_col)
                param_col[fill_spaces] = "nan"
            if param.dtype == "string":
                param_col[fill_spaces] = ""
            whp_param_cols[param] = param_col.astype(dtype_map[param.dtype])
        for param, idx in self.whp_flags.items():
            fill_spaces = np.char.startswith(param_col, "9")
            param_col[fill_spaces] = "nan"
            whp_flag_cols[param] = np_db[:, idx].astype("float16")
        for param, idx in self.whp_errors.items():
            param_col = np_db[:, idx]
            fill_spaces = np.char.startswith(param_col, "-999")
            if param.dtype == "decimal":
                whp_error_precisions[param] = _extract_numeric_precisions(param_col)
                param_col[fill_spaces] = "nan"
            whp_error_cols[param] = param_col.astype(dtype_map[param.dtype])

        comments = "\n".join([*self.stamp, *self.comments])
        del self._raw_lines

        return ExchangeData(
            single_profile,
            whp_param_cols,
            whp_flag_cols,
            whp_error_cols,
            whp_param_precisions,
            whp_error_precisions,
            comments=comments,
        )

    @classmethod
    def from_lines(cls, lines: Tuple[str, ...], ftype: FileType):
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

        class LookingFor(Enum):
            FILE_STAMP = auto()
            COMMENTS = auto()
            CTD_HEADERS = auto()
            PARAMS = auto()
            UNITS = auto()
            DATA = auto()
            POST_DATA = auto()

        state = LookingFor.FILE_STAMP
        ctd_num_headers = 0

        log.debug("Looking for file parts")

        for idx, line in enumerate(lines):
            if state is LookingFor.FILE_STAMP:
                state = LookingFor.COMMENTS
                continue

            if state is LookingFor.COMMENTS:
                if line.startswith("#"):
                    comments_end = idx + 1
                elif ftype == FileType.CTD:
                    state = LookingFor.CTD_HEADERS
                    param, value = _ctd_get_header(line, dtype=int)
                    if param != "NUMBER_HEADERS":
                        raise ValueError()
                    ctd_num_headers = value - 1
                    ctd_header_start = idx + 1
                    continue
                else:
                    state = LookingFor.PARAMS
                    continue

            if state is LookingFor.CTD_HEADERS:
                if ctd_num_headers == 0:
                    ctd_header_end = idx
                    state = LookingFor.PARAMS
                    continue
                ctd_num_headers -= 1

            if state is LookingFor.PARAMS:
                params = idx - 1
                state = LookingFor.UNITS
                continue

            if state is LookingFor.UNITS:
                units = idx - 1
                data_start = idx
                state = LookingFor.DATA
                continue

            if state is LookingFor.DATA:
                if line == "END_DATA":
                    data_end = idx

                    state = LookingFor.POST_DATA
                    post_data_start = post_data_end = idx + 1
                    continue

            if state is LookingFor.POST_DATA:
                post_data_end = idx

        return cls(
            stamp_slice=slice(stamp, comments_start),
            comments_slice=slice(comments_start, comments_end),
            ctd_headers_slice=slice(ctd_header_start, ctd_header_end),
            params_idx=params,
            units_idx=units,
            data_slice=slice(data_start, data_end),
            post_data_slice=slice(post_data_start, post_data_end),
            _raw_lines=lines,
        )


def _extract_numeric_precisions(data: npt.NDArray[np.str_]) -> npt.NDArray[np.int_]:
    """Get the numeric precision of a printed decimal number"""
    # magic number explain: np.char.partition expands each element into a 3-tuple
    # of (pre, sep, post) of some sep, in our case a "." char.
    # We only want the post bits [idx 2] (the number of chars after a decimal seperator)
    # of the last axis.
    numeric_parts = np.char.partition(data, ".")[..., 2]
    str_lens = np.char.str_len(numeric_parts)
    return np.max(str_lens, axis=0)


ExchangeIO = Union[str, Path, io.BufferedIOBase]


def _combine_dt_ndarray(
    date_arr: npt.NDArray[np.str_], time_arr: Optional[npt.NDArray[np.str_]] = None
) -> np.ndarray:
    def _parse_date(dt):
        return datetime.strptime(dt, "%Y%m%d")

    def _parse_datetime(dt):
        return datetime.strptime(dt, "%Y%m%d%H%M")

    # vectorize here doesn't speed things, it just nice for the interface
    parseDate = np.vectorize(_parse_date, ["datetime64"])
    parseDatetime = np.vectorize(_parse_datetime, ["datetime64"])

    if time_arr is None:
        return parseDate(date_arr).astype("datetime64[D]")

    arr = np.char.add(date_arr, time_arr)
    return parseDatetime(arr).astype("datetime64[m]")


def sort_ds(dataset: xr.Dataset) -> xr.Dataset:
    """Sorts the data values in the dataset

    Ensures that profiles are in the following order:
    * Earlier before later (time will increase)
    * Southerly before northerly (latitude will increase)
    * Westerly before easterly (longitude will increase)

    The two xy sorts are esentially tie breakers for when we are missing "time"

    Inside profiles:
    * Shallower before Deeper (pressure will increase)
    """
    # first make sure everything is sorted by pressure
    # this is being done "manually" here becuase xarray only supports 1D sorting
    pressure = dataset.pressure
    sorted_indicies = np.argsort(pressure.values, axis=1)

    for var in dataset.variables:
        # this check ensures that the variable being sorted
        # shares the first two dims as pressure, but allows for more dims past that
        if dataset[var].dims[slice(0, len(pressure.dims))] == pressure.dims:
            dataset[var][:] = np.take_along_axis(
                dataset[var].values, sorted_indicies, axis=1
            )

    # now we can just use the xarray sorting, which only supports 1D
    return dataset.sortby(["time", "latitude", "longitude"])


def combine_dt(dataset: xr.Dataset, id_coord: bool = True) -> xr.Dataset:
    """Combine the exchange style string variables of date and optinally time into a single
    variable containing real datetime objects

    This will remove the time variable if present, and replace then rename the date variable.
    Date is replaced/renamed to maintain variable order in the xr.DataSet
    """
    # TODO: support saying "which" variable to look at

    # date and time want specific attrs whos values have been
    # selected by significant debate
    date = dataset["date"]
    time: Optional[xr.DataArray] = dataset.get(
        "time"
    )  # not be present, this is allowed

    if time is None:
        dt_arr = _combine_dt_ndarray(date.values)
    else:
        dt_arr = _combine_dt_ndarray(date.values, time.values)

    precision = 1 / 24 / 60  # minute as day fraction
    if dt_arr.dtype.name == "datetime64[D]":
        precision = 1

    # TODO: Handle non Coordinate variable combining
    time_var = xr.DataArray(
        dt_arr,
        dims=date.dims,
        attrs={
            "standard_name": "time",
            "axis": "T",
            "whp_name": ["DATE", "TIME"],
            "resolution": precision,
        },
    )
    # if the thing being combined is a coordinate, it may not contain vill values
    time_var.encoding["_FillValue"] = None if id_coord else np.nan
    time_var.encoding["units"] = "days since 1950-01-01T00:00Z"
    time_var.encoding["calendar"] = "gregorian"
    time_var.encoding["dtype"] = "double"

    try:
        del dataset["time"]
    except KeyError:
        pass

    # this is being done in a funny way to retain the variable ordering
    dataset["date"] = time_var
    return dataset.rename({"date": "time"})


def set_axis_attrs(dataset: xr.Dataset) -> xr.Dataset:
    dataset.longitude.attrs["axis"] = "X"
    dataset.latitude.attrs["axis"] = "Y"
    dataset.pressure.attrs["axis"] = "Z"
    dataset.pressure.attrs["positive"] = "down"
    dataset.time.attrs["axis"] = "T"
    return dataset


def _load_raw_exchange(filename_or_obj: ExchangeIO) -> list[str]:
    if isinstance(filename_or_obj, str) and filename_or_obj.startswith("http"):
        log.info("Loading object over http")
        data_raw = io.BytesIO(requests.get(filename_or_obj).content)

    elif isinstance(filename_or_obj, (str, Path)):
        log.info("Loading object from local file path")
        with open(filename_or_obj, "rb") as f:
            data_raw = io.BytesIO(f.read())

    elif isinstance(filename_or_obj, io.BufferedIOBase):
        log.info("Loading object open file object")
        data_raw = io.BytesIO(filename_or_obj.read())

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
    return data


def all_same(ndarr: np.ndarray) -> np.bool_:
    return np.all(ndarr == ndarr.flat[0])


def process_coords() -> Dict[WHPName, xr.DataArray]:
    """There is a special set of variables that make up two types of coordinates.

    CCHDO Indexing coordinates:
    * Expocode
    * Station
    * Cast
    * Sample

    Spatiotemporal coordinates:
    * X: longitude
    * Y: latitude
    * Z: pressure
    * T: the combined date and time
    """


def read_exchange(filename_or_obj: ExchangeIO) -> xr.Dataset:

    data = _load_raw_exchange(filename_or_obj)

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
        ExchangeInfo.from_lines(tuple(df.splitlines()), ftype=ftype).finalize()
        for df in data
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

    def _dataarray_factory(param: WHPName, ctype="data") -> xr.DataArray:
        dtype = dtype_map[param.dtype]
        fill = fills_map[param.dtype]

        if ctype == "flag":
            dtype = dtype_map["integer"]
            fill = fills_map["integer"]

        if param.scope == "profile":
            arr = np.full((N_PROF), fill_value=fill, dtype=dtype)
        if param.scope == "sample":
            arr = np.full((N_PROF, N_LEVELS), fill_value=fill, dtype=dtype)

        attrs = param.get_nc_attrs()

        if ctype == "error":
            attrs = param.get_nc_attrs(error=True)

        if ctype == "flag":
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

        da = xr.DataArray(arr, dims=DIMS[: arr.ndim], attrs=attrs)

        if param.dtype == "string":
            da.encoding["dtype"] = "S1"

        da.encoding["zlib"] = True
        if ctype == "flag":
            da.encoding["dtype"] = "int8"
            da.encoding["_FillValue"] = 9

        return da

    log.debug("Init DataArrays")
    for param in sorted(params):
        dataarrays[param.nc_name] = _dataarray_factory(param)

        if param in flags:
            dataarrays[f"{param.nc_name}_qc"] = _dataarray_factory(param, ctype="flag")

        if param in errors:
            dataarrays[f"{param.nc_name}_error"] = _dataarray_factory(
                param, ctype="error"
            )

    log.debug("Put data in arrays")
    comments = exchange_data[0].comments
    for n_prof, exd in enumerate(exchange_data):
        if exd.comments != comments:
            comments = f"{comments}\n----file_break----\n{exd.comments}"

        for param in params:
            if param in exd.param_precisions:
                dataarrays[param.nc_name].attrs[
                    "source_C_format"
                ] = f"%.{exd.param_precisions[param]}f"
            if param in exd.error_precisions:
                dataarrays[f"{param.nc_name}_error"].attrs[
                    "source_C_format"
                ] = f"%.{exd.error_precisions[param]}f"

            if param.scope == "profile":
                if not all_same(exd.param_cols[param]):
                    raise ExchangeDataInconsistentCoordinateError()
                dataarrays[param.nc_name][n_prof] = exd.param_cols[param][0]

                if param in flags:
                    dataarrays[f"{param.nc_name}_qc"][n_prof] = exd.flag_cols[param][0]
                if param in errors:
                    dataarrays[f"{param.nc_name}_error"][n_prof] = exd.error_cols[
                        param
                    ][0]

            if param.scope == "sample":
                data = exd.param_cols[param]
                dataarrays[param.nc_name][n_prof, : len(data)] = data

                if param in flags:
                    data = exd.flag_cols[param]
                    dataarrays[f"{param.nc_name}_qc"][n_prof, : len(data)] = data
                if param in errors:
                    data = exd.error_cols[param]
                    dataarrays[f"{param.nc_name}_error"][n_prof, : len(data)] = data

    ds = xr.Dataset(
        dataarrays,
        attrs={
            "Conventions": f"CF-1.8 CCHDO-{CCHDO_VERSION}",
            "cchdo_software_version": f"hydro {hydro_version}",
            "cchdo_parameters_version": f"params {params_version}",
            "comments": comments,
            "featureType": "profile",
        },
    )

    # The order of the following is somewhat important
    ds = combine_dt(ds)
    ds = ds.set_coords([coord.nc_name for coord in COORDS if coord.nc_name in ds])
    ds = sort_ds(ds)
    ds = set_axis_attrs(ds)
    ds = add_profile_type(ds, ftype=ftype)
    ds = add_geometry_var(ds)
    return ds
