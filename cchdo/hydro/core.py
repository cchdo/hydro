"""Core operations on a CCHDO CF/netCDF file
"""
from typing import Dict, Hashable, Tuple

import numpy as np
import numpy.typing as npt
import xarray as xr

from cchdo.params import WHPNames, WHPName
from .exchange.flags import ExchangeBottleFlag, ExchangeCTDFlag, ExchangeSampleFlag
from .exchange import (
    combine_dt,
    set_axis_attrs,
    add_profile_type,
    add_geometry_var,
    FileType,
    set_coordinate_encoding_fill,
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

FLAG_SCHEME = {
    "woce_bottle": ExchangeBottleFlag,
    "woce_discrete": ExchangeSampleFlag,
    "woce_ctd": ExchangeCTDFlag,
}


def _dataarray_factory(
    param: WHPName, ctype="data", N_PROF=0, N_LEVELS=0
) -> xr.DataArray:
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

    if param in COORDS and param != CTDPRS:
        var_da.encoding["_FillValue"] = None
        if param.dtype == "integer":
            var_da = var_da.fillna(-999).astype("int32")

    if ctype == "flag":
        var_da.encoding["dtype"] = "int8"
        var_da.encoding["_FillValue"] = 9

    var_da.encoding["zlib"] = True

    return var_da


def add_param(ds: xr.Dataset, param: WHPName, with_flag=False) -> xr.Dataset:
    return ds


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
    new_profs: Dict[Hashable, npt.NDArray] = {
        "expocode": expocode,
        "station": station,
        "cast": cast,
        "time": time.astype("datetime64[ns]"),  # ensure ns precision for now
        "latitude": latitude,
        "longitude": longitude,
        "profile_type": profile_type,
    }

    dataarrays: Dict[Hashable, Tuple[Tuple[Hashable, ...], npt.ArrayLike]] = {}
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
                np.empty((1, ds.dims["N_LEVELS"]), dtype=variable.dtype),
            )
    ds = xr.concat([ds, xr.Dataset(dataarrays)], dim="N_PROF")

    # scalar var is expanded... squish it
    ds["geometry_container"] = ds.geometry_container.squeeze()

    ds = ds.set_coords([coord.nc_name for coord in COORDS if coord.nc_name in ds])
    return ds


def create_new() -> xr.Dataset:
    """Create an empty CF Dataset with the minimum required contents"""
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
