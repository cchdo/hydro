"""Legacy COARDS netcdf make from libcchdo ported to take a CCHDO CF/netCDF xarray.Dataset object as input

The goal is, as much as possible, to use the old code with minimal changes such that the following outputs are identical:

Exchange -> CF/netCDF -> COARDS netCDF (this library)
Exchange -> COARDS netCDF (using libcchdo)
"""
import datetime
from typing import Optional

import numpy as np
import xarray as xr

# TODO put behind a guard
from netCDF4 import Dataset

# These consts are taken directly from libcchdo
QC_SUFFIX = "_QC"
FILE_EXTENSION = "nc"
EPOCH = datetime.datetime(1980, 1, 1, 0, 0, 0)

STATIC_PARAMETERS_PER_CAST = (
    "EXPOCODE",
    "SECT_ID",
    "STNNBR",
    "CASTNO",
    "_DATETIME",
    "LATITUDE",
    "LONGITUDE",
    "DEPTH",
    "BTLNBR",
    "SAMPNO",
)

NON_FLOAT_PARAMETERS = ("NUMBER",)

UNKNOWN = "UNKNOWN"

UNSPECIFIED_UNITS = "unspecified"

STRLEN = 40


# utility functions from libcchdo.formats.woce
def strftime_woce_date(dt: datetime.datetime):
    return dt.strftime("%Y%m%d")


def strftime_woce_time(dt: datetime.datetime):
    return dt.strftime("%H%M")


# new behavior will always have the input be a datetime.datetime (Or np.datetime64)
# this function needs modification
def strftime_woce_date_time(dt):
    if dt is None:
        return (None, None)
    if type(dt) is datetime.date:
        return (strftime_woce_date(dt), None)
    return (strftime_woce_date(dt), strftime_woce_time(dt))


# utility functions from libcchdo.formats.netcdf

# name change ascii -> _ascii to avoid builtin conflict
def _ascii(x: str) -> bytes:
    return x.encode("ascii", "replace")


def simplest_str(s) -> str:
    """Give the simplest string representation.
    If a float is almost equivalent to an integer, swap out for the
    integer.
    """
    # if type(s) is float:
    if isinstance(s, float):
        # if fns.equal_with_epsilon(s, int(s)):
        # replace with equivalent numpy call
        if np.isclose(s, int(s), atol=1e-6):
            s = int(s)
    return str(s)


def _pad_station_cast(x):
    """Pad a station or cast identifier out to 5 characters. This is usually
    for use in a file name.
    Args:
         x - a string to be padded
    """
    return simplest_str(x).rjust(5, "0")


def minutes_since_epoch(dt: Optional[datetime.datetime], error=-9):
    if not dt:
        return error
    if type(dt) is datetime.date:
        dt = datetime.datetime(dt.year, dt.month, dt.day)
    delta = dt - EPOCH
    minutes_in_day = 60 * 24
    minutes_in_seconds = 1.0 / 60
    minutes_in_microseconds = minutes_in_seconds / 1.0e6
    return (
        delta.days * minutes_in_day
        + delta.seconds * minutes_in_seconds
        + delta.microseconds * minutes_in_microseconds
    )


# end utility


def define_dimensions(nc_file: Dataset, length: int):
    """Create NetCDF file dimensions."""
    makeDim = nc_file.createDimension
    makeDim("time", 1)
    makeDim("pressure", length)
    makeDim("latitude", 1)
    makeDim("longitude", 1)
    makeDim("string_dimension", STRLEN)


def define_attributes(
    nc_file: Dataset,
    expocode: str,
    sect_id: str,
    data_type: str,
    stnnbr: str,
    castno: int,
    bottom_depth: int,
):
    nc_file.EXPOCODE = expocode
    nc_file.Conventions = "COARDS/WOCE"
    nc_file.WOCE_VERSION = "3.0"
    nc_file.WOCE_ID = sect_id
    nc_file.DATA_TYPE = data_type
    nc_file.STATION_NUMBER = stnnbr
    nc_file.CAST_NUMBER = castno
    nc_file.BOTTOM_DEPTH_METERS = bottom_depth
    # nc_file.Creation_Time = fns.strftime_iso(datetime.datetime.utcnow())
    nc_file.Creation_Time = datetime.datetime.utcnow().isoformat()


# This will just be the "comment" attribute of the Dataset in CCHDO CF/netCDF
# which will already be formatted correctly
def set_original_header(nc_file: Dataset, ds: xr.Dataset):  # dfile, datatype):
    # nc_file.ORIGINAL_HEADER = '\n'.join([
    #    '{0},{1}'.format(datatype, dfile.globals.get('stamp', '')),
    #    dfile.globals.get('header', '')])
    nc_file.ORIGINAL_HEADER = ds.attrs.get("Comments", "")


def create_common_variables(
    nc_file: Dataset, latitude: float, longitude: float, woce_datetime, stnnbr, castno
):
    """Add variables to the netcdf file object such as date, time etc."""
    # Coordinate variables

    var_time = nc_file.createVariable("time", "i", ("time",))
    var_time.long_name = "time"
    # Java OceanAtlas 5.0.2 requires ISO 8601 with space separator.
    var_time.units = "minutes since %s" % EPOCH.isoformat(" ")
    var_time.data_min = int(minutes_since_epoch(woce_datetime))
    var_time.data_max = var_time.data_min
    var_time.C_format = "%10d"
    var_time[:] = var_time.data_min

    var_latitude = nc_file.createVariable("latitude", "f", ("latitude",))
    var_latitude.long_name = "latitude"
    var_latitude.units = "degrees_N"
    var_latitude.data_min = float(latitude)
    var_latitude.data_max = var_latitude.data_min
    var_latitude.C_format = "%9.4f"
    var_latitude[:] = var_latitude.data_min

    var_longitude = nc_file.createVariable("longitude", "f", ("longitude",))
    var_longitude.long_name = "longitude"
    var_longitude.units = "degrees_E"
    var_longitude.data_min = float(longitude)
    var_longitude.data_max = var_longitude.data_min
    var_longitude.C_format = "%9.4f"
    var_longitude[:] = var_longitude.data_min

    strs_woce_datetime = strftime_woce_date_time(woce_datetime)

    var_woce_date = nc_file.createVariable("woce_date", "i", ("time",))
    var_woce_date.long_name = "WOCE date"
    var_woce_date.units = "yyyymmdd UTC"
    var_woce_date.data_min = int(strs_woce_datetime[0] or -9)
    var_woce_date.data_max = var_woce_date.data_min
    var_woce_date.C_format = "%8d"
    var_woce_date[:] = var_woce_date.data_min

    if strs_woce_datetime[1]:
        var_woce_time = nc_file.createVariable("woce_time", "i2", ("time",))
        var_woce_time.long_name = "WOCE time"
        var_woce_time.units = "hhmm UTC"
        var_woce_time.data_min = int(strs_woce_datetime[1] or -9)
        var_woce_time.data_max = var_woce_time.data_min
        var_woce_time.C_format = "%4d"
        var_woce_time[:] = var_woce_time.data_min

    # Hydrographic specific

    var_station = nc_file.createVariable("station", "c", ("string_dimension",))
    var_station.long_name = "STATION"
    var_station.units = UNSPECIFIED_UNITS
    var_station.C_format = "%s"
    var_station[:] = simplest_str(stnnbr).ljust(len(var_station))

    var_cast = nc_file.createVariable("cast", "c", ("string_dimension",))
    var_cast.long_name = "CAST"
    var_cast.units = UNSPECIFIED_UNITS
    var_cast.C_format = "%s"
    var_cast[:] = simplest_str(castno).ljust(len(var_cast))
