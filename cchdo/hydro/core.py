"""Core operations on a CCHDO CF/netCDF file."""

from collections.abc import Hashable

import numpy as np
import numpy.typing as npt
import xarray as xr

from cchdo.params import WHPName, WHPNames

from .exchange import (
    FileType,
    add_geometry_var,
    add_profile_type,
    combine_dt,
    set_axis_attrs,
    set_coordinate_encoding_fill,
)
from .exchange.flags import (
    ExchangeBottleFlag,
    ExchangeCTDFlag,
    ExchangeFlag,
    ExchangeSampleFlag,
)

DIMS = ("N_PROF", "N_LEVELS")
FILLS_MAP = {"string": "", "integer": np.nan, "decimal": np.nan}
dtype_map = {"string": "U", "integer": "float32", "decimal": "float64"}

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

FLAG_SCHEME: dict[str, type[ExchangeFlag]] = {
    "woce_bottle": ExchangeBottleFlag,
    "woce_discrete": ExchangeSampleFlag,
    "woce_ctd": ExchangeCTDFlag,
}


def _dataarray_factory(
    param: WHPName, ctype="data", N_PROF=0, N_LEVELS=0
) -> xr.DataArray:
    dtype = dtype_map[param.dtype]
    fill = FILLS_MAP[param.dtype]
    name = param.full_nc_name

    if ctype == "flag":
        dtype = dtype_map["integer"]
        fill = FILLS_MAP["integer"]
        name = param.nc_name_flag

    if param.scope == "profile":
        arr = np.full((N_PROF), fill_value=fill, dtype=dtype)
    if param.scope == "sample":
        arr = np.full((N_PROF, N_LEVELS), fill_value=fill, dtype=dtype)

    attrs = param.get_nc_attrs()
    if "C_format" in attrs:
        attrs["C_format_source"] = "database"

    if ctype == "error":
        attrs = param.get_nc_attrs(error=True)
        name = param.nc_name_error

    if ctype == "flag" and param.flag_w in FLAG_SCHEME:
        flag_defs = FLAG_SCHEME[param.flag_w]
        flag_values = []
        flag_meanings = []
        for flag in flag_defs:
            flag_values.append(int(flag))
            flag_meanings.append(flag.cf_def)

        odv_conventions_map = {
            "woce_bottle": "WOCESAMPLE - WOCE Quality Codes for the sampling device itself",
            "woce_ctd": "WOCECTD - WOCE Quality Codes for CTD instrument measurements",
            "woce_discrete": "WOCEBOTTLE - WOCE Quality Codes for water sample (bottle) measurements",
        }

        attrs = {
            "standard_name": "status_flag",
            "flag_values": np.array(flag_values, dtype="int8"),
            "flag_meanings": " ".join(flag_meanings),
            "conventions": odv_conventions_map[param.flag_w],
        }

    var_da = xr.DataArray(arr, dims=DIMS[: arr.ndim], attrs=attrs, name=name)

    if param.dtype == "string":
        var_da.encoding["dtype"] = "S1"

    if param.dtype == "integer":
        var_da.encoding["dtype"] = "int32"
        var_da.encoding["_FillValue"] = -999  # classic

    if param in COORDS and param != CTDPRS:
        var_da.encoding["_FillValue"] = None
        if param.dtype == "integer":
            var_da = var_da.fillna(-999).astype("int32")

    if ctype == "flag":
        var_da.encoding["dtype"] = "int8"
        var_da.encoding["_FillValue"] = 9

    var_da.encoding["zlib"] = True

    return var_da


def add_param(
    ds: xr.Dataset,
    param: WHPName,
    with_flag=False,
    with_error=False,
    with_ancillary=None,
) -> xr.Dataset:
    _ds = ds.copy()
    vars_to_add = []

    if param.full_nc_name in _ds:
        var = _ds[param.full_nc_name]
    else:
        var = _dataarray_factory(
            param, N_PROF=ds.sizes["N_PROF"], N_LEVELS=ds.sizes["N_LEVELS"]
        )
        vars_to_add.append(var)

    if with_flag and param.nc_name_flag not in _ds:
        flag_var = _dataarray_factory(
            param,
            N_PROF=ds.sizes["N_PROF"],
            N_LEVELS=ds.sizes["N_LEVELS"],
            ctype="flag",
        )
        ancillary = var.attrs.get("ancillary_variables", "").split()
        if flag_var.name not in ancillary:
            ancillary.append(flag_var.name)
        var.attrs["ancillary_variables"] = " ".join(ancillary)
        vars_to_add.append(flag_var)

    if with_error and param.full_error_name is None:
        raise ValueError(f"{param} does not have a defined error/uncertainty name")

    if with_error and param.nc_name_error not in _ds:
        error_var = _dataarray_factory(
            param,
            N_PROF=ds.sizes["N_PROF"],
            N_LEVELS=ds.sizes["N_LEVELS"],
            ctype="error",
        )
        ancillary = var.attrs.get("ancillary_variables", "").split()
        if error_var.name not in ancillary:
            ancillary.append(error_var.name)
        var.attrs["ancillary_variables"] = " ".join(ancillary)
        vars_to_add.append(error_var)

    for var in vars_to_add:
        _ds[var.name] = var

    return _ds


def add_profile_level(ds: xr.Dataset, idx, levels) -> xr.Dataset:
    return ds


def add_level(ds: xr.Dataset, n_levels=1) -> xr.Dataset:
    return ds


def add_profile(
    ds: xr.Dataset,
    expocode: npt.ArrayLike,
    station: npt.ArrayLike,
    cast: npt.ArrayLike,
    time: npt.ArrayLike,
    latitude: npt.ArrayLike,
    longitude: npt.ArrayLike,
    profile_type: npt.ArrayLike,
) -> xr.Dataset:
    ds = ds.reset_coords()

    (
        expocode,
        station,
        cast,
        time,
        latitude,
        longitude,
        profile_type,
    ) = np.broadcast_arrays(
        np.atleast_1d(expocode), station, cast, time, latitude, longitude, profile_type
    )
    new_profs: dict[Hashable, npt.NDArray] = {
        "expocode": expocode,
        "station": station,
        "cast": cast,
        "time": time.astype("datetime64[ns]"),  # ensure ns precision for now
        "latitude": latitude,
        "longitude": longitude,
        "profile_type": profile_type,
    }

    dataarrays: dict[Hashable, tuple[tuple[Hashable, ...], npt.ArrayLike]] = {}
    for name, variable in ds.variables.items():
        if name in new_profs:
            data = new_profs[name].astype(variable.dtype.kind)
        if len(variable.dims) == 0:
            dataarrays[name] = (variable.dims, np.nan)
        elif len(variable.dims) == 1:
            dataarrays[name] = (variable.dims, data)
        elif len(variable.dims) == 2:
            dataarrays[name] = (
                variable.dims,
                np.empty((1, ds.sizes["N_LEVELS"]), dtype=variable.dtype),
            )
    ds = xr.concat([ds, xr.Dataset(dataarrays)], dim="N_PROF")

    # scalar var is expanded... squish it
    ds["geometry_container"] = ds.geometry_container.squeeze()

    ds = ds.set_coords([coord.nc_name for coord in COORDS if coord.nc_name in ds])
    return ds


def create_new() -> xr.Dataset:
    """Create an empty CF Dataset with the minimum required contents."""
    dataarrays = {}
    for param in COORDS:
        dataarrays[param.nc_name] = _dataarray_factory(param)
    ds = xr.Dataset(dataarrays)

    ds = set_coordinate_encoding_fill(ds)

    ds = combine_dt(ds)

    ds = ds.set_coords([coord.nc_name for coord in COORDS if coord.nc_name in ds])

    ds = add_profile_type(ds, FileType.BOTTLE)  # just adds the var if no dims > 0
    ds = set_axis_attrs(ds)
    ds = add_geometry_var(ds)
    return ds
