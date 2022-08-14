import logging
import io
import dataclasses
from typing import Any, Set, Tuple, Dict, Union, Optional, Iterable, List, TypedDict
from operator import attrgetter
from functools import cached_property
from itertools import chain
from warnings import warn
from zipfile import ZipFile, is_zipfile
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum, auto

from typing_extensions import TypeGuard  # move to stdlib when min ver is 3.10

import requests
import numpy as np
import numpy.typing as npt

import xarray as xr

from cchdo.params import WHPName, WHPNames
from cchdo.params._version import version as params_version

from .exceptions import (
    ExchangeDataFlagPairError,
    ExchangeDataInconsistentCoordinateError,
    ExchangeDataPartialCoordinateError,
    ExchangeDataPartialKeyError,
    ExchangeDuplicateKeyError,
    ExchangeEncodingError,
    ExchangeBOMError,
    ExchangeError,
    ExchangeInconsistentMergeType,
    ExchangeMagicNumberError,
    ExchangeDuplicateParameterError,
    ExchangeParameterUnitAlignmentError,
    ExchangeOrphanFlagError,
    ExchangeOrphanErrorError,
    ExchangeParameterUndefError,
    ExchangeFlaglessParameterError,
    ExchangeFlagUnitError,
)
from .flags import ExchangeBottleFlag, ExchangeCTDFlag, ExchangeSampleFlag

try:
    from .. import __version__ as hydro_version

    CCHDO_VERSION = ".".join(hydro_version.split(".")[:2])
    if "dev" in hydro_version:
        CCHDO_VERSION = hydro_version
except ImportError:
    hydro_version = CCHDO_VERSION = "unknown"

__all__ = ["read_exchange"]

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
BTLNBR = WHPNames["BTLNBR"]

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

FILLS_MAP = {"string": "", "integer": np.nan, "decimal": np.nan}


class FileType(Enum):
    CTD = "C"
    BOTTLE = "B"


# WHPNameIndex represents a Name to Column index in an exchange file
WHPNameIndex = Dict[WHPName, int]
# WHPParamUnit represents the paired up contents of the Parameter and Unit lines
# in an exchange file
WHPParamUnit = Tuple[str, Optional[str]]


def _bottle_get_params(params_units: Iterable[WHPParamUnit]) -> WHPNameIndex:
    """Given an ordered iterable of param, unit pairs, return the index of the column in the datafile for known WHP params.

    Exchange files have comma separated parameter names on one line, and the corresponding units on the next.
    This function will search for this name+unit pair in the builtin database of known WHP parameter names and return a mapping of :py:class:`~hydro.data.WHPName` to column indicies.

    It is currently an error for the parameter in a file to not be in the built in database.

    This function will ignore uncertainty (error) columns and flag columns, those are parsed by other functions.

    .. warning::

        Convert semantically empty units (e.g. empty string, all whitespace) to None before passing into this function

    .. note::

        The parameter name database will convert unambiguous aliases to their canonical exchange parameter and unit pair.

    :param params_units: Paired (e.g. zip) parameter names and units
    :type params_units: tuple in the form (str, str) or (str, None)
    :returns: Mapping of :py:class:`~hydro.data.WHPName` to column indicies
    :rtype: dict with keys of :py:class:`~hydro.data.WHPName` and values of int
    :raises ExchangeParameterUndefError: if the parameter unit pair cannot be found in the built in database
    """
    params = {}
    unknown_errors = []
    duplicate_errors = []
    for index, (param, unit) in enumerate(params_units):
        if (param, unit) in WHPNames.error_cols:
            continue
        if param.endswith("_FLAG_W"):
            continue
        try:
            # TODO: remove ignore type error when upstream is fixed
            whpname = WHPNames[(param, unit)]  # type: ignore
        except KeyError:
            unknown_errors.append((param, unit))
        if whpname in params:
            duplicate_errors.append(whpname)
        params[whpname] = index  # type: ignore

    if any(unknown_errors):
        raise ExchangeParameterUndefError(unknown_errors)

    if any(duplicate_errors):
        raise ExchangeDuplicateParameterError(
            f"The following params are duplicate: {duplicate_errors}"
        )
    return params


def _bottle_get_flags(
    params_units: Iterable[WHPParamUnit], whp_params: WHPNameIndex
) -> WHPNameIndex:
    """Given an ordered iterable of param unit pairs and WHPNames known to be in the file, return the index of the column indicies of the flags for the WHPNames.

    Exchange files can have status flags for some of the parameters.
    Flag columns must have no units.
    Some parameters must not have status flags, these include the spatiotemporal parameters (e.g. lat, lon, but also pressure) and the sample identifying parameters (expocode, station, cast, sample, but *not* bottle id).

    :param params_units: Paired (e.g. zip) parameter names and units
    :type params_units: tuple in the form (str, str) or (str, None)
    :param whp_params: Mapping of parameters known to be in the file, this is the output of :py:func:`._bottle_get_params`
    :type whp_params: Mapping of :py:class:`~hydro.data.WHPName` to int
    :returns: Mapping of :py:class:`~hydro.data.WHPName` to column indicies for the status flag column
    :rtype: dict with keys of :py:class:`~hydro.data.WHPName` and values of int
    :raises ExchangeFlagUnitError: if the flag column has units other than None
    :raises ExchangeFlaglessParameterError: if the flag column is for a parameter not allowed to have status flags
    :raises ExchangeOrphanFlagError: if the flag column is for a parameter not in the passed in mapping of whp_params
    """
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
    params_units: Iterable[WHPParamUnit], whp_params: WHPNameIndex
) -> WHPNameIndex:
    """Given an ordered iterable of param unit pairs and WHPNames known to be in the file, return the index of the column indicies of the errors/uncertanties for the WHPNames.

    Some parameters may have uncertanties associated with them, this function finds those columns and pairs them with the correct parameter.

    .. note::

        There is no programable way to find the error columns for a given unit (e.g. no common suffix like the flags).
        This must be done via lookup in the built in database of params.

    :param params_units: Paired (e.g. :py:func:`zip`) parameter names and units
    :type params_units: tuple in the form (str, str) or (str, None)
    :param whp_params: Mapping of parameters known to be in the file, this is the output of :py:func:`._bottle_get_params`
    :type whp_params: Mapping of :py:class:`~hydro.data.WHPName` to int
    :returns: Mapping of :py:class:`~hydro.data.WHPName` to column indicies for the error column
    :rtype: dict with keys of :py:class:`~hydro.data.WHPName` and values of int
    """
    param_errs = {}

    for index, (param, unit) in enumerate(params_units):
        if (param, unit) not in WHPNames.error_cols:
            continue

        for name in whp_params.keys():
            if name.error_name == param and name.whp_unit == unit:
                param_errs[name] = index

    return param_errs


def _ctd_get_header(line, dtype=str):
    header, value = (part.strip() for part in line.split("="))
    if header in ("_SAMPLING_RATE", "SAMPLING_RATE") and value.lower().endswith("hz"):
        value = value.rstrip(" HZhz")
    return header, dtype(value)


def _is_all_dataarray(val: List[Any]) -> TypeGuard[List[xr.DataArray]]:
    return all(isinstance(obj, xr.DataArray) for obj in val)


def add_cdom_coordinate(dataset: xr.Dataset) -> xr.Dataset:
    """Find all the paraters in the cdom group and add their wavelength in a new coordinate"""

    # this needs to be a set to deal with the potential for aliasesd names
    cdom_names = {
        name.nc_name
        for name in filter(lambda x: x.nc_group == "cdom", WHPNames.values())
    }

    # done in a way the preserves the order of the params and QC flags in the dataset
    cdom_data = [
        dataarray for dataarray in dataset.values() if dataarray.name in cdom_names
    ]

    # nothing to do
    if len(cdom_data) == 0:
        return dataset

    cdom_qc = [
        dataset.get(dataarray.attrs.get("ancillary_variables"))
        for dataarray in cdom_data
    ]

    # useful for later coping of attrs
    first = cdom_data[0]

    # "None in" doesn't seem to work due to xarray comparison?
    none_in_qc = [da is None for da in cdom_qc]
    if any(none_in_qc) and not all(none_in_qc):
        raise NotImplementedError("partial QC for CDOM is not handled yet")

    radiation_wavelengths = []
    for dataarray in cdom_data:
        whp_name = dataarray.attrs["whp_name"]
        whp_unit = dataarray.attrs["whp_unit"]
        whpname = WHPNames[(whp_name, whp_unit)]
        radiation_wavelengths.append(whpname.radiation_wavelength)

    cdom_wavelengths = xr.DataArray(
        np.array(radiation_wavelengths, dtype=np.int32),
        dims="CDOM_WAVELENGTHS",
        name="CDOM_WAVELENGTHS",
        attrs={
            "standard_name": "radiation_wavelength",
            "units": "nm",
        },
    )

    new_cdom_dims = ("N_PROF", "N_LEVELS", "CDOM_WAVELENGTHS")
    new_cdom_coords = {
        "N_PROF": first.coords["N_PROF"],
        "N_LEVELS": first.coords["N_LEVELS"],
        "CDOM_WAVELENGTHS": cdom_wavelengths,
    }

    has_qc = False
    # qc flags first if any
    if _is_all_dataarray(cdom_qc):
        has_qc = True
        cdom_qc_arrays = np.stack(cdom_qc, axis=-1)
        first_qc = cdom_qc[0]

        new_cdom_qc_attrs = {**first_qc.attrs}
        new_cdom_qc = xr.DataArray(
            cdom_qc_arrays,
            dims=new_cdom_dims,
            coords=new_cdom_coords,
            attrs=new_cdom_qc_attrs,
        )
        new_cdom_qc.encoding = first_qc.encoding

    cdom_arrays = np.stack(cdom_data, axis=-1)

    new_cdom_attrs = {**first.attrs, "whp_name": "CDOM{CDOM_WAVELENGTHS}"}

    new_qc_name = f"{whpname.nc_group}_qc"

    if has_qc:
        new_cdom_attrs["ancillary_variables"] = new_qc_name

    new_cdom = xr.DataArray(
        cdom_arrays, dims=new_cdom_dims, coords=new_cdom_coords, attrs=new_cdom_attrs
    )
    new_cdom.encoding = first.encoding

    dataset[first.name] = new_cdom
    dataset = dataset.rename({first.name: whpname.nc_group})

    if has_qc:
        dataset[first_qc.name] = new_cdom_qc
        dataset = dataset.rename({first_qc.name: new_qc_name})

    for old_name in cdom_names:
        try:
            del dataset[old_name]
        except KeyError:
            pass

    for old_name in cdom_names:
        try:
            del dataset[f"{old_name}_qc"]
        except KeyError:
            pass

    return dataset


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
                set(ancillary_variables)
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

    if BTL_DATE.nc_name not in dataset and BTL_TIME.nc_name not in dataset:
        return dataset

    if BTL_TIME.nc_name in dataset and BTL_DATE.nc_name not in dataset:
        dates = np.char.replace(
            np.datetime_as_string(dataset[TIME.nc_name].values, unit="D"), "-", ""
        )

        dataset[BTL_DATE.nc_name] = dataset[BTL_TIME.nc_name].copy()
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


def check_flags(dataset: xr.Dataset, raises=True):
    """Check WOCE flag values agaisnt their param and ensure that the param either has a value or is "nan"
    depedning on the flag definition.

    Return a boolean array of invalid locations?
    """
    woce_flags = {
        "WOCESAMPLE": ExchangeBottleFlag,
        "WOCECTD": ExchangeCTDFlag,
        "WOCEBOTTLE": ExchangeSampleFlag,
    }
    flag_has_value = {
        "WOCESAMPLE": {flag.value: flag.has_value for flag in ExchangeBottleFlag},
        "WOCECTD": {flag.value: flag.has_value for flag in ExchangeCTDFlag},
        "WOCEBOTTLE": {flag.value: flag.has_value for flag in ExchangeSampleFlag},
    }
    # In some cases, a coordinate variable might have flags, so we are not using filter_by_attrs
    # get all the flag vars (that also have conventions)
    flag_vars = []
    for var_name in dataset.variables:
        # do not replace the above with .items() it will give you xr.Variable objects that you don't want to use
        # the following gets a real xr.DataArray
        data = dataset[var_name]
        if not {"standard_name", "conventions"} <= data.attrs.keys():
            continue
        if not any(flag in data.attrs["conventions"] for flag in woce_flags):
            continue
        if "status_flag" in data.attrs["standard_name"]:
            flag_vars.append(var_name)

    # match flags with their data vars
    # it is legal in CF for one set of flags to apply to multiple vars
    flag_errors = {}
    for flag_var in flag_vars:

        # get the flag and check attrs for defs
        flag_da = dataset[flag_var]
        conventions = None
        for flag in woce_flags:
            if flag_da.attrs.get("conventions", "").startswith(flag):
                conventions = flag
                break

        # we don't know these flags, skip the check
        if not conventions:
            continue

        allowed_values = np.array(list(flag_has_value[conventions]))
        illegal_flags = ~flag_da.fillna(9).isin(allowed_values)
        if np.any(illegal_flags):
            illegal_flags.attrs[
                "comments"
            ] = f"This is a boolean array in the same shape as '{flag_da.name}' which is truthy where invalid values exist"
            flag_errors[f"{flag_da.name}_value_errors"] = illegal_flags
            continue

        for var_name in dataset.variables:
            data = dataset[var_name]
            if "ancillary_variables" not in data.attrs:
                continue
            if flag_var not in data.attrs["ancillary_variables"].split(" "):
                continue

            # check data against flags
            has_fill_f = [
                flag
                for flag, value in flag_has_value[conventions].items()
                if value is False
            ]

            has_fill = flag_da.isin(has_fill_f) | np.isnan(flag_da)

            # TODO deal with strs

            if np.issubdtype(data.values.dtype, np.number):
                fill_value_mismatch: xr.DataArray = ~(np.isfinite(data) ^ has_fill)  # type: ignore # numpy doesn't support __array_ufunc__ types yet
                if np.any(fill_value_mismatch):
                    fill_value_mismatch.attrs[
                        "comments"
                    ] = f"This is a boolean array in the same shape as '{data.name}' which is truthy where invalid values exist"
                    flag_errors[f"{data.name}_value_errors"] = fill_value_mismatch

    flag_errors_ds = xr.Dataset(flag_errors)
    if raises and any(flag_errors_ds):
        raise ExchangeDataFlagPairError(flag_errors_ds)

    return flag_errors_ds


@dataclasses.dataclass
class _ExchangeData:
    """Dataclass containing exchange data which has been parsed into ndarrays"""

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
        self, params: Set[WHPName], flags: Set[WHPName], errors: Set[WHPName]
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
    def str_lens(self) -> Dict[WHPName, int]:
        """Figure out the length of all the string params

        The char size can vary by platform.
        """
        np_char_size = np.dtype("U1").itemsize
        lens = {}
        for param, data in self.param_cols.items():
            if param.dtype == "string":
                lens[param] = data.itemsize // np_char_size

        return lens


def _get_fill_locs(arr, fill_values: Tuple[str, ...] = ("-999",)):
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
    _raw_lines: Tuple[str, ...] = dataclasses.field(repr=False)

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
        return [c[1:] if c.startswith("#") else c for c in raw_comments]

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
    def whp_params(self):
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

        params_idx = _bottle_get_params(zip(self.params, self.units))

        if any(self.ctd_headers):
            params_idx[SAMPNO] = params_idx[CTDPRS]

        return params_idx

    @cached_property
    def whp_flags(self):
        """Parses the params and units for flag values

        returns a dict with a WHPName to column index of flags mapping
        """
        return _bottle_get_flags(zip(self.params, self.units), self.whp_params)

    @cached_property
    def whp_errors(self):
        """Parses the params and units for uncertanty values

        returns a dict with a WHPName to column index of errors mapping
        """
        return _bottle_get_errors(zip(self.params, self.units), self.whp_params)

    @property
    def _np_data_block(self):
        _raw_data = tuple(
            tuple((*self.ctd_headers.values(), *line.replace(" ", "").split(",")))
            for line in self.data
        )
        return np.array(_raw_data, dtype="U")

    def finalize(self, fill_values=("-999",), precision_source="file") -> _ExchangeData:
        """Parse all the data into ndarrays of the correct dtype and shape

        Returns an ExchangeData dataclass
        """
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
                    raise ValueError("exchange numeric data has bad chars")
                if precision_source == "file":
                    whp_param_precisions[param] = _extract_numeric_precisions(param_col)
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
                if precision_source == "file":
                    whp_error_precisions[param] = _extract_numeric_precisions(param_col)
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


def _is_valid_exchange_numeric(data: npt.NDArray[np.str_]) -> np.bool_:
    # see allowed code points of the exchange doc
    # essentially, only %f types (not %g)
    allowed_exchange_numeric_data_chars = [
        c.encode("utf-8") for c in list("0123456789.-")
    ] + [b""]
    aligned = np.require(data, requirements=["C_CONTIGUOUS"])
    return np.all(np.isin(aligned.view("|S1"), allowed_exchange_numeric_data_chars))


ExchangeIO = Union[str, Path, io.BufferedIOBase]


def _combine_dt_ndarray(
    date_arr: npt.NDArray[np.str_],
    time_arr: Optional[npt.NDArray[np.str_]] = None,
    time_pad=False,
) -> np.ndarray:

    # TODO: When min pyver is 3.10, maybe consider pattern matching here
    def _parse_date(date_val: str) -> np.datetime64:
        if date_val == "":
            return np.datetime64("nat")
        return np.datetime64(datetime.strptime(date_val, "%Y%m%d"))

    def _parse_datetime(date_val: str) -> np.datetime64:
        if date_val == "T":
            return np.datetime64("nat")
        if date_val.endswith("2400"):
            date, _ = date_val.split("T")
            return np.datetime64(datetime.strptime(date, "%Y%m%d") + timedelta(days=1))
        return np.datetime64(datetime.strptime(date_val, "%Y%m%dT%H%M"))

    # vectorize here doesn't speed things, it just nice for the interface
    parse_date = np.vectorize(_parse_date, ["datetime64"])
    parse_datetime = np.vectorize(_parse_datetime, ["datetime64"])

    if time_arr is None:
        return parse_date(date_arr).astype("datetime64[D]")

    if np.all(time_arr == "0"):
        return parse_date(date_arr).astype("datetime64[D]")

    if time_pad:
        if np.any(np.char.str_len(time_arr[time_arr != ""]) < 4):
            warn("Time values are being padded with zeros")
        time_arr[time_arr != ""] = np.char.zfill(time_arr[time_arr != ""], 4)

    arr = np.char.add(np.char.add(date_arr, "T"), time_arr)
    return parse_datetime(arr).astype("datetime64[m]")


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


def check_sorted(dataset: xr.Dataset) -> bool:
    """Check that the dataset is sorted by the rules in :py:`sort_ds`"""
    sorted_ds = sort_ds(dataset.copy(deep=True))

    return all(
        [
            np.allclose(sorted_ds.pressure, dataset.pressure, equal_nan=True),
            np.all(
                (sorted_ds.time == dataset.time)
                | (np.isnat(sorted_ds.time) == np.isnat(dataset.time))
            ),
            np.allclose(sorted_ds.latitude, dataset.latitude, equal_nan=True),
            np.allclose(sorted_ds.longitude, dataset.longitude, equal_nan=True),
        ]
    )


WHPNameAttr = Union[str, List[str]]


def combine_dt(
    dataset: xr.Dataset,
    is_coord: bool = True,
    date_name: WHPName = DATE,
    time_name: WHPName = TIME,
    time_pad=False,
) -> xr.Dataset:
    """Combine the exchange style string variables of date and optinally time into a single
    variable containing real datetime objects

    This will remove the time variable if present, and replace then rename the date variable.
    Date is replaced/renamed to maintain variable order in the xr.DataSet
    """

    # date and time want specific attrs whos values have been
    # selected by significant debate
    date = dataset[date_name.nc_name]
    time: Optional[xr.DataArray] = dataset.get(
        time_name.nc_name
    )  # not be present, this is allowed

    whp_name: WHPNameAttr = [date_name.whp_name, time_name.whp_name]
    try:
        if time is None:
            dt_arr = _combine_dt_ndarray(date.values)
        else:
            dt_arr = _combine_dt_ndarray(date.values, time.values, time_pad=time_pad)
    except ValueError as err:
        raise ExchangeError(
            f"Could not parse date/time cols {date_name.whp_name} {time_name.whp_name}"
        ) from err

    precision = 1 / 24 / 60  # minute as day fraction
    if dt_arr.dtype.name == "datetime64[D]":
        precision = 1
        whp_name = date_name.whp_name

    time_var = xr.DataArray(
        dt_arr,
        dims=date.dims,
        attrs={
            "standard_name": "time",
            "whp_name": whp_name,
            "resolution": precision,
        },
    )
    if is_coord is True:
        time_var.attrs["axis"] = "T"

    # if the thing being combined is a coordinate, it may not contain vill values
    time_var.encoding["_FillValue"] = None if is_coord else np.nan
    time_var.encoding["units"] = "days since 1950-01-01T00:00Z"
    time_var.encoding["calendar"] = "gregorian"
    time_var.encoding["dtype"] = "double"

    try:
        del dataset[time_name.nc_name]
    except KeyError:
        pass

    # this is being done in a funny way to retain the variable ordering
    # we will always keep the "time" variable name
    dataset[date_name.nc_name] = time_var
    return dataset.rename({date_name.nc_name: time_name.nc_name})


def set_axis_attrs(dataset: xr.Dataset) -> xr.Dataset:
    """Set the CF axis attribute on our axis variables (XYZT)

    * longitude = "X"
    * latitude = "Y"
    * pressure = "Z", addtionally, positive is down
    * time = "T"
    """
    dataset.longitude.attrs["axis"] = "X"
    dataset.latitude.attrs["axis"] = "Y"
    dataset.pressure.attrs["axis"] = "Z"
    dataset.pressure.attrs["positive"] = "down"
    dataset.time.attrs["axis"] = "T"
    return dataset


def set_coordinate_encoding_fill(dataset: xr.Dataset) -> xr.Dataset:
    """Sets the _FillValue encoidng to None for 1D coordinate vars"""
    for coord in COORDS:
        if coord is TIME and coord.nc_name not in dataset:
            continue
        if len(dataset[coord.nc_name].dims) == 1:
            dataset[coord.nc_name].encoding["_FillValue"] = None

    return dataset


def _load_raw_exchange(filename_or_obj: ExchangeIO) -> List[str]:
    if isinstance(filename_or_obj, str) and filename_or_obj.startswith("http"):
        log.info("Loading object over http")
        data_raw = io.BytesIO(requests.get(filename_or_obj).content)

    elif isinstance(filename_or_obj, (str, Path)) and Path(filename_or_obj).exists():
        log.info("Loading object from local file path")
        with open(filename_or_obj, "rb") as local_file:
            data_raw = io.BytesIO(local_file.read())

    # lets just try "reading"
    elif hasattr(filename_or_obj, "read"):
        log.info("Loading object open file object")
        # https://github.com/python/mypy/issues/1424
        data_raw = io.BytesIO(filename_or_obj.read())  # type: ignore

    elif isinstance(filename_or_obj, (bytes, bytearray)):
        log.info("Loading raw data bytes")
        data_raw = io.BytesIO(filename_or_obj)

    data = []
    if is_zipfile(data_raw):

        data_raw.seek(0)  # is_zipfile moves the "tell" position
        with ZipFile(data_raw) as zipfile:
            for zipinfo in zipfile.infolist():
                log.debug("Reading %s", zipinfo)
                try:
                    data.append(zipfile.read(zipinfo).decode("utf8"))
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
    """Test if all the values of an ndarray are the same value"""
    if np.issubdtype(ndarr.dtype, np.number) and np.isnan(ndarr.flat[0]):
        return np.all(np.isnan(ndarr))
    return np.all(ndarr == ndarr.flat[0])


class CheckOptions(TypedDict, total=False):
    flags: bool


def read_exchange(
    filename_or_obj: ExchangeIO,
    fill_values=("-999",),
    checks: Optional[CheckOptions] = None,
    precision_source="file",
) -> xr.Dataset:
    """Loads the data from filename_or_obj and returns a xr.Dataset with the CCHDO
    CF/netCDF structure"""

    _checks: CheckOptions = {"flags": True}
    if checks is not None:
        _checks.update(checks)

    log.debug(f"Check options: {_checks}")

    data = _load_raw_exchange(filename_or_obj)

    log.info("Checking for BOM")
    if any((df.startswith("\ufeff") for df in data)):
        raise ExchangeBOMError

    log.info("Detecting file type")
    if all((df.startswith("BOTTLE") for df in data)):
        ftype = FileType.BOTTLE
    elif all((df.startswith("CTD") for df in data)):
        ftype = FileType.CTD
    elif all((df.startswith(("CTD", "BOTTLE")) for df in data)):
        # Mixed CTD and BOTTLE files (probably in a zip)
        raise ExchangeInconsistentMergeType
    else:
        raise ExchangeMagicNumberError

    log.info("Found filetype: %s", ftype.name)

    exchange_data = [
        _ExchangeInfo.from_lines(tuple(df.splitlines()), ftype=ftype).finalize(
            fill_values=fill_values,
            precision_source=precision_source,
        )
        for df in data
    ]

    if not all((fp.single_profile for fp in exchange_data)):
        exchange_data = list(chain(*[exd.split_profiles() for exd in exchange_data]))

    N_PROF = len(exchange_data)
    N_LEVELS = max((fp.shape[0] for fp in exchange_data))

    log.debug((N_PROF, N_LEVELS))

    params = set(chain(*[exd.param_cols.keys() for exd in exchange_data]))
    flags = set(chain(*[exd.flag_cols.keys() for exd in exchange_data]))
    errors = set(chain(*[exd.error_cols.keys() for exd in exchange_data]))
    for exd in exchange_data:
        exd.set_expected(params, flags, errors)

    log.debug("Dealing with strings")
    str_len = 1
    for exd in exchange_data:
        for param, value in exd.str_lens.items():
            str_len = max(value, str_len)

    dataarrays = {}
    dtype_map = {"string": f"U{str_len}", "integer": "float32", "decimal": "float64"}

    def _dataarray_factory(param: WHPName, ctype="data") -> xr.DataArray:
        dtype = dtype_map[param.dtype]
        fill = FILLS_MAP[param.dtype]

        if ctype == "flag":
            dtype = dtype_map["integer"]
            fill = FILLS_MAP["integer"]

        if param.scope == "profile":
            arr = np.full((N_PROF), fill_value=fill, dtype=dtype)
        if param.scope == "sample":
            arr = np.full((N_PROF, N_LEVELS), fill_value=fill, dtype=dtype)

        attrs = param.get_nc_attrs()
        if "C_format" in attrs:
            attrs["C_format_source"] = "database"

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

        var_da = xr.DataArray(arr, dims=DIMS[: arr.ndim], attrs=attrs)

        if param.dtype == "string":
            var_da.encoding["dtype"] = "S1"

        if param.dtype == "integer":
            var_da.encoding["dtype"] = "int32"
            var_da.encoding["_FillValue"] = -999  # classic

        if param in COORDS:
            var_da.encoding["_FillValue"] = None
            if param.dtype == "integer":
                var_da = var_da.fillna(-999).astype("int32")

        if ctype == "flag":
            var_da.encoding["dtype"] = "int8"
            var_da.encoding["_FillValue"] = 9

        var_da.encoding["zlib"] = True

        return var_da

    log.debug("Init DataArrays")
    for param in sorted(params):
        dataarrays[param.nc_name] = _dataarray_factory(param)

        dataarrays[param.nc_name].attrs["ancillary_variables"] = []
        if param in flags:
            qc_name = f"{param.nc_name}_qc"
            dataarrays[qc_name] = _dataarray_factory(param, ctype="flag")
            dataarrays[param.nc_name].attrs["ancillary_variables"].append(qc_name)

        if param in errors:
            error_name = f"{param.nc_name}_error"
            dataarrays[error_name] = _dataarray_factory(param, ctype="error")
            dataarrays[param.nc_name].attrs["ancillary_variables"].append(error_name)

        # Check for ancillary temperature data and connect to the parent
        if param.analytical_temperature_name is not None:
            ancilary_temp_param = WHPNames[
                (param.analytical_temperature_name, param.analytical_temperature_units)
            ]
            if ancilary_temp_param in params:
                dataarrays[param.nc_name].attrs["ancillary_variables"].append(
                    ancilary_temp_param.nc_name
                )

    log.debug("Put data in arrays")
    comments = exchange_data[0].comments
    for n_prof, exd in enumerate(exchange_data):
        if exd.comments != comments:
            comments = f"{comments}\n----file_break----\n{exd.comments}"

        for param in params:
            if param in exd.param_precisions:
                dataarrays[param.nc_name].attrs[
                    "C_format"
                ] = f"%.{exd.param_precisions[param]}f"
                dataarrays[param.nc_name].attrs["C_format_source"] = "input_file"
            if param in exd.error_precisions:
                dataarrays[f"{param.nc_name}_error"].attrs[
                    "C_format"
                ] = f"%.{exd.error_precisions[param]}f"
                dataarrays[f"{param.nc_name}_error"].attrs[
                    "C_format_source"
                ] = "input_file"

            if param.scope == "profile":
                if not all_same(exd.param_cols[param]):
                    raise ExchangeDataInconsistentCoordinateError(param)
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
        [coord.nc_name for coord in COORDS if coord.nc_name in ex_dataset]
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
