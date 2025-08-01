import dataclasses
import io
import logging
from collections.abc import Iterable
from enum import Enum, auto
from functools import cached_property
from itertools import chain
from operator import attrgetter
from pathlib import Path
from typing import (
    TypedDict,
    TypeGuard,
)
from zipfile import ZipFile, is_zipfile

import numpy as np
import numpy.typing as npt
import requests
import xarray as xr

from cchdo.hydro.checks import check_flags
from cchdo.hydro.consts import (
    BTLNBR,
    CASTNO,
    COORDS,
    CTDPRS,
    EXPOCODE,
    FILLS_MAP,
    LATITUDE,
    SAMPNO,
    STNNBR,
    TIME,
)
from cchdo.hydro.core import dataarray_factory
from cchdo.hydro.dt import combine_dt
from cchdo.hydro.exchange.exceptions import (
    ExchangeBOMError,
    ExchangeDataInconsistentCoordinateError,
    ExchangeDataPartialCoordinateError,
    ExchangeDataPartialKeyError,
    ExchangeDuplicateKeyError,
    ExchangeDuplicateParameterError,
    ExchangeEncodingError,
    ExchangeFlaglessParameterError,
    ExchangeInconsistentMergeType,
    ExchangeMagicNumberError,
    ExchangeOrphanErrorError,
    ExchangeOrphanFlagError,
    ExchangeParameterUndefError,
    ExchangeParameterUnitAlignmentError,
)
from cchdo.hydro.flags import (
    ExchangeBottleFlag,
    ExchangeCTDFlag,
    ExchangeFlag,
    ExchangeSampleFlag,
)
from cchdo.hydro.sorting import sort_ds
from cchdo.hydro.types import (
    FileType,
    FileTypeType,
    PrecisionSource,
    PrecisionSourceType,
)
from cchdo.hydro.utils import (
    add_cdom_coordinate,
    add_geometry_var,
    add_profile_type,
    all_same,
    extract_numeric_precisions,
    set_axis_attrs,
    set_coordinate_encoding_fill,
)
from cchdo.params import WHPName, WHPNames
from cchdo.params import __version__ as params_version

try:
    from cchdo.hydro import __version__ as hydro_version

    CCHDO_VERSION = ".".join(hydro_version.split(".")[:2])
    if "dev" in hydro_version:
        CCHDO_VERSION = hydro_version
except ImportError:
    hydro_version = CCHDO_VERSION = "unknown"

log = logging.getLogger(__name__)


SENTINEL_PREFIX = "__SENTINEL"
SENTINEL_PARAM = WHPName(
    SENTINEL_PREFIX, SENTINEL_PREFIX, SENTINEL_PREFIX, -1, "string", False, 0
)

FLAG_SCHEME: dict[str, type[ExchangeFlag]] = {
    "woce_bottle": ExchangeBottleFlag,
    "woce_discrete": ExchangeSampleFlag,
    "woce_ctd": ExchangeCTDFlag,
}


# WHPNameIndex represents a Name to Column index in an exchange file
WHPNameIndex = dict[WHPName, int]
# WHPParamUnit represents the paired up contents of the Parameter and Unit lines
# in an exchange file
WHPParamUnit = tuple[str, str | None]


def _has_no_nones(val: list[str | None]) -> TypeGuard[list[str]]:
    return None not in val


def _transform_whp_to_csv(params: list[str], units: list[str]) -> list[str]:
    slots: list[str | None] = [None for _ in range(len(params))]

    pairs = list(zip(params, units, strict=True))
    mutable_units = list(units)

    if len(set(pairs)) != len(pairs):
        # we will assume that this is due to flags, actual duplciate params will come later
        for index, (param, unit) in enumerate(pairs):
            next_idx = index + 1
            try:
                next_param, next_unit = pairs[next_idx]
            except IndexError:
                continue

            potential_flag = f"{param}_FLAG_W"
            if next_param == potential_flag:
                mutable_units[next_idx] = unit

    flags = {}
    # param pass
    for index, (param, unit) in enumerate(zip(params, mutable_units, strict=True)):
        if param.endswith("_FLAG_W"):
            flags[param] = index
        slots[index] = f"{param} [{unit}]"

    # flag pass
    for param, unit in zip(params, mutable_units, strict=True):
        if param.endswith("_FLAG_W"):
            continue
        if (flag := f"{param}_FLAG_W") in flags:
            slots[flags[flag]] = f"{param} [{unit}]_FLAG_W"

    if _has_no_nones(slots):
        return slots

    raise ValueError("something has gone wrong with parameters transform")


def _get_params(
    params_units: Iterable[str], ignore_params: Iterable[str] | None = None
) -> tuple[WHPNameIndex, WHPNameIndex, WHPNameIndex]:
    params: WHPNameIndex = {}
    flags: WHPNameIndex = {}
    errors: WHPNameIndex = {}

    duplicate_errors = []
    unknown_errors = []

    if ignore_params is None:
        ignore_params = ()

    for index, param in enumerate(params_units):
        if param.startswith("__SENTINEL") or param in ignore_params:
            continue

        try:
            whpname = WHPNames[param]
        except KeyError:
            unknown_errors.append(param)
            continue

        if whpname.error_col:
            errors[whpname] = index

        elif whpname.flag_col:
            flags[whpname] = index

        else:
            if whpname in params:
                duplicate_errors.append(param)

            params[whpname] = index

    if any(unknown_errors):
        raise ExchangeParameterUndefError(unknown_errors)

    if any(duplicate_errors):
        raise ExchangeDuplicateParameterError(
            f"The following params are duplicate: {duplicate_errors}"
        )

    if not (params.keys() >= flags.keys()):
        raise ValueError(f"Some flags not in params: {flags.keys() - params.keys()}")
    if not (params.keys() >= errors.keys()):
        raise ValueError(f"Some errors not in params: {errors.keys() - params.keys()}")

    for flag in flags:
        if flag.flag_w == "no_flags":
            raise ExchangeFlaglessParameterError(flag)

    return params, flags, errors


def _ctd_get_header(line, dtype=str):
    header, value = (part.strip() for part in line.split("="))
    if header in ("_SAMPLING_RATE", "SAMPLING_RATE") and value.lower().endswith("hz"):
        value = value.rstrip(" HZhz")
    return header, dtype(value)


def finalize_ancillary_variables(dataset: xr.Dataset):
    """Turn the ancillary variable attr into a space seperated string

    It is nice to have the ancillary variable be a list while things are being read into it
    """
    for var in dataset.variables:
        if "ancillary_variables" not in dataset[var].attrs:
            continue
        ancillary_variables = dataset[var].attrs["ancillary_variables"]
        if len(ancillary_variables) == 0:
            del dataset[var].attrs["ancillary_variables"]
        elif isinstance(ancillary_variables, str):
            pass
        elif isinstance(ancillary_variables, list):
            dataset[var].attrs["ancillary_variables"] = " ".join(
                sorted(set(ancillary_variables))
            )
        else:
            raise ValueError("ancillary variables are crazy")

    return dataset


def combine_bottle_time(dataset: xr.Dataset):
    """Combine the bottle dates and times if present

    Raises if only one is present
    """
    BTL_TIME = WHPNames["BTL_TIME"]
    BTL_DATE = WHPNames["BTL_DATE"]

    if BTL_DATE.full_nc_name not in dataset and BTL_TIME.full_nc_name not in dataset:
        return dataset

    if BTL_TIME.nc_name in dataset and BTL_DATE.nc_name not in dataset:
        dates = np.char.replace(
            np.datetime_as_string(dataset[TIME.nc_name].values, unit="D"), "-", ""
        )

        dataset[BTL_DATE.nc_name] = dataset[BTL_TIME.nc_name].copy().astype("U8")
        dataset[BTL_DATE.nc_name].values.T[:] = dates
        dataset[BTL_DATE.nc_name].values[dataset[BTL_TIME.nc_name].values == ""] = ""

    ds = combine_dt(
        dataset,
        is_coord=False,
        date_name=BTL_DATE,
        time_name=BTL_TIME,
        time_pad=True,
    )

    # Take the station time as BO and go back one hour for "safty"
    reference_time = ds.time - np.timedelta64(1, "h")

    # Add a day to anything before the ref time
    next_day = (ds.bottle_time < reference_time).values
    ds.bottle_time.values[next_day] = ds.bottle_time.values[next_day] + np.timedelta64(
        1, "D"
    )

    return ds


def check_is_subset_shape(
    a1: npt.NDArray, a2: npt.NDArray, strict="disallowed"
) -> npt.NDArray[np.bool_]:
    """Ensure that the shape of the data in a2 is a subset (or strict subset) of the data shape of a1

    For a given set of param, flag, and error arrays you would want to ensure that:

    * errors are a subset of params (strict is allowed)
    * params are a subset of flags (strict is allowed)

    For string vars, the empty string is considered the "nothing" value.
    For woce flags, flag 9s should be converted to nans (depending on scheme flag 5 and 1 may not have param values)

    Return a boolean array of invalid locations
    """
    if a1.shape != a2.shape:
        raise ValueError("Cannot compare diffing shaped arrays")

    a1_values = np.isfinite(a1)
    a2_values = np.isfinite(a2)

    return a1_values != a2_values


@dataclasses.dataclass
class _ExchangeData:
    """Dataclass containing exchange data which has been parsed into ndarrays"""

    single_profile: bool
    param_cols: dict[WHPName, np.ndarray]
    flag_cols: dict[WHPName, np.ndarray]
    error_cols: dict[WHPName, np.ndarray]

    # OG Print Precition Tracking
    param_precisions: dict[WHPName, npt.NDArray[np.int_]]
    error_precisions: dict[WHPName, npt.NDArray[np.int_]]

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
        if not all(shape == shapes[0] for shape in shapes):
            # TODO Error handling
            raise ValueError("shape error")

        self.shape = shapes[0]

        if self.single_profile:
            # all "profile scoped" params must have the same values
            for param, data in self.param_cols.items():
                if param.scope != "profile":
                    continue
                if not np.unique(data).shape[0] == 1:
                    raise ValueError(f"inconsistent {param} {data}")

            # sample must be unique
            try:
                sample_ids = self.param_cols[SAMPNO]
            except KeyError as err:
                log.debug("SAMPNO not in file, attempting BTLNBR fallback")
                if BTLNBR in self.param_cols:
                    sample_ids = self.param_cols[BTLNBR]
                    self.param_cols[SAMPNO] = self.param_cols[BTLNBR]
                else:
                    raise ExchangeDataPartialKeyError("Missing SAMPNO") from err

            unique_sample_ids, unique_sample_counts = np.unique(
                sample_ids, return_counts=True
            )
            if unique_sample_ids.shape != sample_ids.shape:
                duplicated_values = unique_sample_ids[unique_sample_counts > 1]
                raise ExchangeDuplicateKeyError(
                    {
                        "EXPOCODE": self.param_cols[EXPOCODE][0],
                        "STNNBR": self.param_cols[STNNBR][0],
                        "CASTNO": self.param_cols[CASTNO][0],
                        "SAMPNO": str(duplicated_values),
                    }
                )

            # check coordinates are "full"
            for coord in COORDS:
                if coord is TIME and TIME not in self.param_cols:
                    continue
                data = self.param_cols[coord]
                if data.dtype.char in {"S", "U"}:
                    if np.any(data == ""):
                        raise ExchangeDataPartialCoordinateError(
                            f"{coord} has missing values"
                        )
                elif np.any(np.isnan(data)):
                    raise ExchangeDataPartialCoordinateError(
                        f"{coord} has missing values"
                    )

        # make sure flags and errors are strict subsets
        if not self.flag_cols.keys() <= self.param_cols.keys():
            raise ExchangeOrphanFlagError()
        if not self.error_cols.keys() <= self.param_cols.keys():
            raise ExchangeOrphanErrorError()

    def set_expected(
        self, params: set[WHPName], flags: set[WHPName], errors: set[WHPName]
    ):
        """Puts fill columns for expected params which are missing

        This can occur when there are disjoint columns in CTD files
        """
        ref_cols = {
            "string": EXPOCODE,
            "integer": CASTNO,
            "decimal": LATITUDE,
        }

        # we need to detect if just the flag is misisng and set to flag 0 or 9 depending on where data are
        # else set to flag 9
        for name in flags:
            if name in self.flag_cols:
                continue
            self.flag_cols[name] = np.full_like(
                self.param_cols[ref_cols["integer"]], fill_value=np.nan
            )
            if name in self.param_cols:
                self.flag_cols[name][np.isfinite(self.param_cols[name])] = 0

        for name in params:
            if name in self.param_cols:
                continue
            self.param_cols[name] = np.full_like(
                self.param_cols[ref_cols[name.dtype]], fill_value=FILLS_MAP[name.dtype]
            )

        for name in errors:
            if name in self.error_cols:
                continue
            self.error_cols[name] = np.full_like(
                self.param_cols[ref_cols[name.dtype]], fill_value=FILLS_MAP[name.dtype]
            )

    def split_profiles(self):
        """Split into single profile containing _ExchangeData instances

        Done by looking at the expocode+station+cast composate keys
        """
        try:
            expocode = self.param_cols[EXPOCODE]
        except KeyError as err:
            raise ExchangeDataPartialKeyError("Missing EXPOCODE") from err
        try:
            station = self.param_cols[STNNBR]
        except KeyError as err:
            raise ExchangeDataPartialKeyError("Missing STNNBR") from err
        try:
            cast = self.param_cols[CASTNO]
        except KeyError as err:
            raise ExchangeDataPartialKeyError("Missing CASTNO") from err

        # need to split up by profiles and _not_ assume the bottles are in order
        # use the actual values to sort things out
        # we don't care what the values are, they just need to work
        log.debug("Grouping Profiles by Key")
        # we need to add seperators to avoid conflicts
        # TODO: add test for when these might conflict
        expocode_sep = np.char.add(expocode, ",")
        station_sep = np.char.add(station, ",")
        # numpy concat basically
        prof_ids = np.char.add(np.char.add(expocode_sep, station_sep), cast.astype("U"))
        unique_profile_ids = np.unique(prof_ids)
        log.debug("Found %s unique profile keys", len(unique_profile_ids))
        profiles = [np.nonzero(prof_ids == prof) for prof in unique_profile_ids]

        log.debug("Actually splitting profiles")
        return [
            _ExchangeData(
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

    @cached_property
    def str_lens(self) -> dict[WHPName, int]:
        """Figure out the length of all the string params

        The char size can vary by platform.
        """
        log.debug("Dealing with strings")
        lens = {}
        for param, data in self.param_cols.items():
            if param.dtype == "string":
                lens[param] = int(np.max(np.char.str_len(data)))

        return lens


def _get_fill_locs(arr, fill_values: tuple[str, ...] = ("-999",)):
    fill = np.char.startswith(arr, fill_values[0])
    if len(fill_values) > 1:
        for fill_value in fill_values[1:]:
            fill = fill | np.char.startswith(arr, fill_value)
    return fill


@dataclasses.dataclass
class _ExchangeInfo:
    """Low level dataclass containing the parts of an exchange file"""

    stamp_slice: slice
    comments_slice: slice
    ctd_headers_slice: slice
    params_idx: int
    units_idx: int
    data_slice: slice
    post_data_slice: slice
    _raw_lines: tuple[str, ...] = dataclasses.field(repr=False)
    _ctd_override: bool = False
    _ignore_columns: Iterable[str] = ()

    @property
    def stamp(self):
        """Returns the filestamp of the exchange file

        e.g. "BOTTLE,20210301CCHSIOAMB"
        """
        return self._raw_lines[self.stamp_slice]

    @property
    def comments(self):
        """Returns the comments of the exchange file with leading # stripped"""
        raw_comments = self._raw_lines[self.comments_slice]
        return [c.removeprefix("#") for c in raw_comments]

    @property
    def ctd_headers(self):
        """Returns a dict of the CTD headers and their value"""
        return dict(
            [_ctd_get_header(line) for line in self._raw_lines[self.ctd_headers_slice]]
        )

    @cached_property
    def params(self):
        """Returns a list of all parameters in the file (including CTD "headers")"""
        ctd_params = self.ctd_headers.keys()
        data_params = self._raw_lines[self.params_idx].split(",")
        return [param.strip() for param in [*ctd_params, *data_params]]

    @cached_property
    def units(self):
        """Returns a list of all the units in the file (including CTD "headers")

        Will have the same shape as params
        """
        if self.params_idx == self.units_idx:
            return self.params

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
        """Returns the data block of an exchange file as a tuple of strs.
        One line per entry.
        """
        return self._raw_lines[self.data_slice]

    @property
    def post_data(self):
        """Returns any post data content as a tuple of strs"""
        return self._raw_lines[self.post_data_slice]

    @cached_property
    def _whp_param_info(self):
        """Parses the params and units for base parameters

        Returns a dict with a WHPName to column index mapping
        """
        # In initial testing, it was discovered that approx half the ctd files
        # had trailing commas in just the params and units lines
        if self.params[-1] == "" and self.units[-1] is None:
            self.params.pop()
            self.units.pop()

        # the number of expected columns is just going to be the number of
        # parameter names we see
        column_count = len(self.params)

        if len(self.units) != column_count:
            if len(self.units) > column_count:
                # attempt to fix trailing commas in units (assume PARAMS is canonical)
                while len(self.units) > column_count:
                    if self.units[-1] is not None:
                        break
                    self.units.pop()

            # check to see if above fixed it
            if len(self.units) != column_count:
                raise ExchangeParameterUnitAlignmentError

        if self.params_idx == self.units_idx:
            params_units = self.params
        else:
            params_units = _transform_whp_to_csv(self.params, self.units)
        params_idx, flags, errors = _get_params(params_units, self._ignore_columns)

        if any(self.ctd_headers) or self._ctd_override:
            params_idx[SAMPNO] = params_idx[CTDPRS]

        return params_idx, flags, errors

    @property
    def whp_params(self):
        return self._whp_param_info[0]

    @property
    def whp_flags(self):
        """Parses the params and units for flag values

        returns a dict with a WHPName to column index of flags mapping
        """
        return self._whp_param_info[1]

    @property
    def whp_errors(self):
        """Parses the params and units for uncertanty values

        returns a dict with a WHPName to column index of errors mapping
        """
        return self._whp_param_info[2]

    @property
    def _np_data_block(self):
        _raw_data = tuple(
            (*self.ctd_headers.values(), *line.replace(" ", "").split(","))
            for line in self.data
        )
        return np.array(_raw_data, dtype="U")

    def finalize(
        self,
        fill_values=("-999",),
        precision_source: PrecisionSourceType = PrecisionSource.FILE,
    ) -> _ExchangeData:
        """Parse all the data into ndarrays of the correct dtype and shape

        Returns an ExchangeData dataclass
        """
        precision_source = PrecisionSource(precision_source)

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
            fill_spaces = _get_fill_locs(param_col, fill_values)
            if param.dtype in ("decimal", "integer"):
                if not _is_valid_exchange_numeric(param_col):
                    raise ValueError(
                        f"exchange numeric data for {param.whp_name} has bad chars"
                    )
                if precision_source == PrecisionSource.FILE:
                    whp_param_precisions[param] = extract_numeric_precisions(param_col)
                param_col[fill_spaces] = "nan"
            if param.dtype == "string":
                param_col[fill_spaces] = ""
            whp_param_cols[param] = param_col.astype(dtype_map[param.dtype])

        for param, idx in self.whp_flags.items():
            param_col = np_db[:, idx]
            fill_spaces = np.char.startswith(param_col, "9")
            param_col[fill_spaces] = "nan"
            whp_flag_cols[param] = np_db[:, idx].astype("float16")

        for param, idx in self.whp_errors.items():
            param_col = np_db[:, idx]
            fill_spaces = _get_fill_locs(param_col, fill_values)
            if param.dtype in ("decimal", "integer"):
                if not _is_valid_exchange_numeric(param_col):
                    raise ValueError(
                        f"{param} error col exchange numeric data has bad chars"
                    )
                if precision_source == PrecisionSource.FILE:
                    whp_error_precisions[param] = extract_numeric_precisions(param_col)
                param_col[fill_spaces] = "nan"
            whp_error_cols[param] = param_col.astype(dtype_map[param.dtype])

        comments = "\n".join([*self.stamp, *self.comments])
        del self._raw_lines

        return _ExchangeData(
            single_profile,
            whp_param_cols,
            whp_flag_cols,
            whp_error_cols,
            whp_param_precisions,
            whp_error_precisions,
            comments=comments,
        )

    @classmethod
    def from_lines(cls, lines: tuple[str, ...], ftype: FileTypeType, ignore_columns):
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

        ftype = FileType(ftype)

        class LookingFor(Enum):
            """States for the FSM that is this parser"""

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
            _ignore_columns=ignore_columns,
        )


def _is_valid_exchange_numeric(data: npt.NDArray[np.str_]) -> np.bool_:
    # see allowed code points of the exchange doc
    # essentially, only %f types (not %g)
    allowed_exchange_numeric_data_chars = [
        c.encode("utf-8") for c in list("0123456789.-")
    ] + [b""]
    aligned = np.require(data, requirements=["C_CONTIGUOUS"])
    return np.all(np.isin(aligned.view("|S1"), allowed_exchange_numeric_data_chars))


ExchangeIO = str | Path | io.BufferedIOBase


def _load_raw_exchange(
    filename_or_obj: ExchangeIO,
    *,
    file_seperator: str | None = None,
    keep_seperator=True,
    encoding="utf8",
) -> list[str]:
    if isinstance(filename_or_obj, str) and filename_or_obj.startswith("http"):
        log.info("Loading object over http")
        data_raw = io.BytesIO(requests.get(filename_or_obj).content)

    elif isinstance(filename_or_obj, str | Path) and Path(filename_or_obj).exists():
        log.info("Loading object from local file path")
        with open(filename_or_obj, "rb") as local_file:
            data_raw = io.BytesIO(local_file.read())

    # lets just try "reading"
    elif hasattr(filename_or_obj, "read"):
        log.info("Loading object open file object")
        data_raw = io.BytesIO(filename_or_obj.read())

    elif isinstance(filename_or_obj, bytes | bytearray):
        log.info("Loading raw data bytes")
        data_raw = io.BytesIO(filename_or_obj)

    data: list[str] = []

    if file_seperator is not None:
        data = data_raw.read().decode(encoding).strip().split(file_seperator)
        data = list(filter(lambda x: x != "", data))

        if keep_seperator:
            data = [(datum + file_seperator).strip() for datum in data]

        data = [datum.strip() for datum in data]

    elif is_zipfile(data_raw):
        data_raw.seek(0)  # is_zipfile moves the "tell" position
        with ZipFile(data_raw) as zipfile:
            for zipinfo in zipfile.infolist():
                log.debug("Reading %s", zipinfo)
                try:
                    data.append(zipfile.read(zipinfo).decode(encoding))
                except UnicodeDecodeError as error:
                    raise ExchangeEncodingError from error
    else:
        data_raw.seek(0)  # is_zipfile moves the "tell" position
        try:
            data.append(data_raw.read().decode(encoding))
        except UnicodeDecodeError as error:
            raise ExchangeEncodingError from error

    # cleanup the data_raw to free the memory
    data_raw.close()
    return data


class CheckOptions(TypedDict, total=False):
    """Flags and config that controll how strict the file checks are"""

    flags: bool


def read_csv(
    filename_or_obj: ExchangeIO,
    *,
    fill_values=("-999",),
    ftype: FileTypeType = FileType.BOTTLE,
    checks: CheckOptions | None = None,
    precision_source: PrecisionSourceType = PrecisionSource.FILE,
    encoding="utf8",
    ignore_columns: Iterable[str] | None = None,
) -> xr.Dataset:
    precision_source = PrecisionSource(precision_source)
    ftype = FileType(ftype)

    _checks: CheckOptions = {"flags": True}
    if checks is not None:
        _checks.update(checks)

    if ignore_columns is None:
        ignore_columns = ()

    data = _load_raw_exchange(
        filename_or_obj,
        file_seperator="something_very_unliekly~~~",
        keep_seperator=False,
        encoding=encoding,
    )

    if len(data) != 1:
        raise ValueError("read_csv can only read a single file")

    splitdata = data[0].splitlines()

    params_units = splitdata[0]
    whp_params: list[WHPName] = []

    sentinel_count = 0
    for name in params_units.split(","):
        if name in ignore_columns:
            whp_params.append(SENTINEL_PARAM.as_depth(sentinel_count + 1))
            sentinel_count += 1
        else:
            whp_params.append(WHPNames[name])

    params_units_list = [name.odv_key for name in whp_params]

    NONE_SLICE = slice(
        0,
        0,
    )
    new_data = (",".join(params_units_list), *splitdata[1:])
    exchange_data = _ExchangeInfo(
        stamp_slice=NONE_SLICE,
        comments_slice=NONE_SLICE,
        ctd_headers_slice=NONE_SLICE,
        params_idx=0,
        units_idx=0,
        data_slice=slice(1, None),
        post_data_slice=NONE_SLICE,
        _raw_lines=new_data,
        _ctd_override=ftype == FileType.CTD,
    ).finalize(precision_source=precision_source, fill_values=fill_values)
    return _from_exchange_data([exchange_data], ftype=ftype, checks=_checks)


def read_exchange(
    filename_or_obj: ExchangeIO,
    *,
    fill_values=("-999",),
    checks: CheckOptions | None = None,
    precision_source: PrecisionSourceType = PrecisionSource.FILE,
    file_seperator=None,
    keep_seperator=True,
    encoding="utf8",
    ignore_columns: Iterable[str] | None = None,
) -> xr.Dataset:
    """Loads the data from filename_or_obj and returns a xr.Dataset with the CCHDO
    CF/netCDF structure"""

    precision_source = PrecisionSource(precision_source)

    _checks: CheckOptions = {"flags": True}
    if checks is not None:
        _checks.update(checks)

    log.debug(f"Check options: {_checks}")

    data = _load_raw_exchange(
        filename_or_obj,
        file_seperator=file_seperator,
        keep_seperator=keep_seperator,
        encoding=encoding,
    )

    log.info("Checking for BOM")
    if any(df.startswith("\ufeff") for df in data):
        raise ExchangeBOMError

    log.info("Detecting file type")
    if all(df.startswith("BOTTLE") for df in data):
        ftype = FileType.BOTTLE
    elif all(df.startswith("CTD") for df in data):
        ftype = FileType.CTD
    elif all(df.startswith(("CTD", "BOTTLE")) for df in data):
        # Mixed CTD and BOTTLE files (probably in a zip)
        raise ExchangeInconsistentMergeType
    else:
        raise ExchangeMagicNumberError

    log.info("Found filetype: %s", ftype.name)

    exchange_data = [
        _ExchangeInfo.from_lines(
            tuple(df.splitlines()), ftype=ftype, ignore_columns=ignore_columns
        ).finalize(
            fill_values=fill_values,
            precision_source=precision_source,
        )
        for df in data
    ]

    return _from_exchange_data(exchange_data, ftype=ftype, checks=_checks)


def _from_exchange_data(
    exchange_data: list[_ExchangeData],
    *,
    ftype: FileTypeType = FileType.BOTTLE,
    checks: CheckOptions | None = None,
) -> xr.Dataset:
    _checks: CheckOptions = {"flags": True}
    if checks is not None:
        _checks.update(checks)

    if not all(fp.single_profile for fp in exchange_data):
        exchange_data = list(chain(*[exd.split_profiles() for exd in exchange_data]))

    N_PROF = len(exchange_data)
    N_LEVELS = max(fp.shape[0] for fp in exchange_data)

    log.debug((N_PROF, N_LEVELS))

    params = set(chain(*[exd.param_cols.keys() for exd in exchange_data]))
    flags = set(chain(*[exd.flag_cols.keys() for exd in exchange_data]))
    errors = set(chain(*[exd.error_cols.keys() for exd in exchange_data]))
    for exd in exchange_data:
        exd.set_expected(params, flags, errors)

    # to init the empty data arrays, we need to know the max size of all the string type parameters
    # otherwise the values will be silently truncated
    str_lens: dict[WHPName, int] = {}
    for exd in exchange_data:
        for param, length in exd.str_lens.items():
            if str_lens.get(param, 0) <= length:
                str_lens[param] = length
    log.debug(f"Total string lengths: {str_lens}")

    log.debug("Init DataArrays")
    dataarrays = {}
    for param in sorted(params):
        dataarrays[param.full_nc_name] = dataarray_factory(
            param, N_PROF=N_PROF, N_LEVELS=N_LEVELS, strlen=str_lens.get(param)
        )

        dataarrays[param.full_nc_name].attrs["ancillary_variables"] = []
        if param in flags:
            qc_name = param.nc_name_flag
            dataarrays[qc_name] = dataarray_factory(
                param, ctype="flag", N_PROF=N_PROF, N_LEVELS=N_LEVELS
            )
            dataarrays[param.full_nc_name].attrs["ancillary_variables"].append(qc_name)

        if param in errors:
            error_name = param.nc_name_error
            dataarrays[error_name] = dataarray_factory(
                param, ctype="error", N_PROF=N_PROF, N_LEVELS=N_LEVELS
            )
            dataarrays[param.full_nc_name].attrs["ancillary_variables"].append(
                error_name
            )

        # Check for ancillary temperature data and connect to the parent
        if param.analytical_temperature_name is not None:
            ancilary_temp_param = WHPNames[
                (param.analytical_temperature_name, param.analytical_temperature_units)
            ]
            if ancilary_temp_param in params:
                dataarrays[param.full_nc_name].attrs["ancillary_variables"].append(
                    ancilary_temp_param.full_nc_name
                )

    log.debug("Put data in arrays")
    comments = exchange_data[0].comments
    for n_prof, exd in enumerate(exchange_data):
        if exd.comments != comments:
            comments = f"{comments}\n----file_break----\n{exd.comments}"

        for param in params:
            if param in exd.param_precisions and param.dtype == "decimal":
                dataarrays[param.full_nc_name].attrs["C_format"] = (
                    f"%.{exd.param_precisions[param]}f"
                )
                dataarrays[param.full_nc_name].attrs["C_format_source"] = "input_file"
            if param in exd.error_precisions and param.dtype == "decimal":
                dataarrays[param.nc_name_error].attrs["C_format"] = (
                    f"%.{exd.error_precisions[param]}f"
                )
                dataarrays[param.nc_name_error].attrs["C_format_source"] = "input_file"

            if param.scope == "profile":
                if not all_same(exd.param_cols[param]):
                    raise ExchangeDataInconsistentCoordinateError(param)
                dataarrays[param.full_nc_name][n_prof] = exd.param_cols[param][0]

                if param in flags:
                    dataarrays[param.nc_name_flag][n_prof] = exd.flag_cols[param][0]
                if param in errors:
                    dataarrays[param.nc_name_error][n_prof] = exd.error_cols[param][0]

            if param.scope == "sample":
                data = exd.param_cols[param]
                dataarrays[param.full_nc_name][n_prof, : len(data)] = data

                if param in flags:
                    data = exd.flag_cols[param]
                    dataarrays[param.nc_name_flag][n_prof, : len(data)] = data
                if param in errors:
                    data = exd.error_cols[param]
                    dataarrays[param.nc_name_error][n_prof, : len(data)] = data

    ex_dataset = xr.Dataset(
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
    ex_dataset = set_coordinate_encoding_fill(ex_dataset)
    ex_dataset = combine_dt(ex_dataset, time_pad=True)

    # these are the only two we know of for now
    ex_dataset = ex_dataset.set_coords(
        [coord.full_nc_name for coord in COORDS if coord.full_nc_name in ex_dataset]
    )
    ex_dataset = sort_ds(ex_dataset)
    ex_dataset = set_axis_attrs(ex_dataset)
    ex_dataset = add_profile_type(ex_dataset, ftype=ftype)
    ex_dataset = add_geometry_var(ex_dataset)
    ex_dataset = finalize_ancillary_variables(ex_dataset)
    ex_dataset = combine_bottle_time(ex_dataset)
    ex_dataset = add_cdom_coordinate(ex_dataset)

    if _checks["flags"]:
        log.debug("Checking flags")
        check_flags(ex_dataset)

    return ex_dataset
