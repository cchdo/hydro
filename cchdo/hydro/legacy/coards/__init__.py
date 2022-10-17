"""Legacy COARDS netcdf make from libcchdo ported to take a CCHDO CF/netCDF xarray.Dataset object as input

The goal is, as much as possible, to use the old code with minimal changes such that the following outputs are identical:

Exchange -> CF/netCDF -> COARDS netCDF (this library)
Exchange -> COARDS netCDF (using libcchdo)
"""
import datetime
from csv import DictReader
from logging import getLogger

# TODO: switch to files().joinpath().open when python 3.8 is dropped
# 2023-04-16
from importlib.resources import open_text

import numpy as np
import xarray as xr

# TODO put behind a guard
from netCDF4 import Dataset
from cftime import date2num

from .. import woce

log = getLogger(__name__)

# Load the WHP to nc name mapping
# this was dumped from old libcchdo
with open_text("cchdo.hydro.legacy.coards", "name_netcdf.csv") as params:
    reader = DictReader(params)
    PARAMS = {}
    for param in reader:
        PARAMS[f"{param['name']} [{param['mnemonic']}]"] = param


# These consts are taken directly from libcchdo
FILL_VALUE = -999.0

QC_SUFFIX = b"_QC"
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
    # extra from porting
    ["DATE", "TIME"],
    "DATE",
    "TIME",
)

NON_FLOAT_PARAMETERS = ("CTDNOBS",)

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
def strftime_woce_date_time(dt: xr.DataArray):
    pydate = dt.values.astype("<M8[ms]").astype(datetime.datetime).item()
    if dt is None:
        return (None, None)
    if dt.attrs.get("resolution", 0) >= 1:
        return (strftime_woce_date(pydate), None)
    return (strftime_woce_date(pydate), strftime_woce_time(pydate))


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


def minutes_since_epoch(dt: xr.DataArray, epoch, error=-9):
    return date2num(
        dt.values.astype("<M8[ms]").astype(datetime.datetime),
        epoch,
        calendar="proleptic_gregorian",
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
    expocode,
    sect_id,
    data_type,
    stnnbr,
    castno,
    bottom_depth,
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
    nc_file.Creation_Time = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


# This will just be the "comment" attribute of the Dataset in CCHDO CF/netCDF
# which will already be formatted correctly
def set_original_header(nc_file: Dataset, ds: xr.Dataset):  # dfile, datatype):
    # nc_file.ORIGINAL_HEADER = '\n'.join([
    #    '{0},{1}'.format(datatype, dfile.globals.get('stamp', '')),
    #    dfile.globals.get('header', '')])
    nc_file.ORIGINAL_HEADER = ds.attrs.get("comments", "")


def create_common_variables(
    nc_file: Dataset, latitude: float, longitude: float, woce_datetime, stnnbr, castno
):
    """Add variables to the netcdf file object such as date, time etc."""
    # Coordinate variables

    var_time = nc_file.createVariable("time", "i", ("time",))
    var_time.long_name = "time"
    # Java OceanAtlas 5.0.2 requires ISO 8601 with space separator.
    var_time.units = "minutes since %s" % EPOCH.isoformat(" ")
    var_time.data_min = int(minutes_since_epoch(woce_datetime, var_time.units))
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
    var_station[:] = simplest_str(stnnbr.item()).ljust(len(var_station))

    var_cast = nc_file.createVariable("cast", "c", ("string_dimension",))
    var_cast.long_name = "CAST"
    var_cast.units = UNSPECIFIED_UNITS
    var_cast.C_format = "%s"
    var_cast[:] = simplest_str(castno.item()).ljust(len(var_cast))


# This one will be hard, needs emulation of the legacy params interface
def create_and_fill_data_variables(nc_file, ds: xr.Dataset):
    """Add variables to the netcdf file object that correspond to data."""
    for variable_name, variable in (
        ds.reset_coords().filter_by_attrs(whp_name=lambda x: x is not None).items()
    ):
        parameter_name = variable.attrs["whp_name"]
        parameter_unit = variable.attrs.get("whp_unit", "")
        parameter_key = f"{parameter_name} [{parameter_unit}]"

        # Hacks to match behavior of previous
        if parameter_name == "INSTRUMENT_ID":
            continue

        if parameter_name in STATIC_PARAMETERS_PER_CAST:
            continue

        if parameter_name in NON_FLOAT_PARAMETERS:
            continue

        parameter = PARAMS.get(parameter_key, {})

        pname = parameter.get("name_netcdf")
        if not pname:
            log.warn(
                f"No netcdf name for {parameter_key}. Using mnemonic {parameter_name}."
            )
            pname = parameter_name
        if not pname:
            raise AttributeError("No name found for {0}".format(parameter_name))
        pname = _ascii(pname)

        # XXX HACK
        if pname == b"oxygen1":
            pname = b"oxygen"

        var = nc_file.createVariable(pname, "f8", ("pressure",))
        var.long_name = pname

        if var.long_name == "pressure":
            var.positive = "down"

        units = UNSPECIFIED_UNITS
        if parameter.get("units_name"):
            units = parameter["units_name"]
        elif parameter_unit != "":
            units = parameter_unit
        var.units = _ascii(units)

        # since this function always operates on single profiles with the N_PROF
        # dim still present, we always want the first and only profile
        data = variable.to_numpy()[0]

        # I think this is just checking if there are any holes in the data
        # xarray will basically gaurantee holes will be "nan" for numeric
        if data.dtype.kind in "iuf":
            var.data_min = np.min(data)
            var.data_max = np.max(data)
            if np.any(np.isnan(var.data_min)):
                var.data_min = float("-inf")
                var.data_max = float("inf")

        if parameter.get("format"):
            var.C_format = _ascii(parameter["format"])
        else:
            # TODO TEST this
            log.warn(
                "Parameter {0} has no format. defaulting to '%f'".format(parameter_name)
            )
            var.C_format = b"%f"
        if var.C_format.endswith(b"s"):
            log.warn(
                "Parameter {0} does not have a format string acceptable for "
                "numeric data. Defaulting to '%f' to prevent ncdump "
                "segfault.".format(parameter_name)
            )
            var.C_format = b"%f"
        var.WHPO_Variable_Name = parameter_name
        var[:] = data

        if f"{variable_name}_qc" in ds:
            qc_name = pname + QC_SUFFIX
            var.OBS_QC_VARIABLE = qc_name
            vfw = nc_file.createVariable(qc_name, "i2", ("pressure",))
            vfw.long_name = qc_name + b"_flag"
            vfw.units = "woce_flags"
            vfw.C_format = "%1d"
            vfw[:] = ds[f"{variable_name}_qc"]


def ctd_create_common_variables(nc_file, ds):
    # Lon and lat are now guaranteed
    latitude = float(ds.latitude)
    longitude = float(ds.longitude)
    stnnbr = ds.get("station", UNKNOWN)
    castno = ds.get("cast", UNKNOWN)

    create_common_variables(nc_file, latitude, longitude, ds.time, stnnbr, castno)


def write_ctd(ds: xr.Dataset) -> bytes:
    """How to write a CTD NetCDF file."""
    # When libcchdo was first written, netCDF for python didn't support in memory data
    nc_file = Dataset("inmemory.nc", "w", format="NETCDF3_CLASSIC", memory=0)

    define_dimensions(nc_file, ds.dims["N_LEVELS"])

    # Define dataset attributes
    define_attributes(
        nc_file,
        ds.expocode,  # self.globals.get('EXPOCODE', UNKNOWN),
        ds.get("section_id", UNKNOWN),  # self.globals.get('SECT_ID', UNKNOWN),
        "WOCE CTD",
        ds.station,  # self.globals.get('STNNBR', UNKNOWN),
        ds["cast"].astype(str),  # self.globals.get('CASTNO', UNKNOWN),
        int(
            ds.get("btm_depth", FILL_VALUE)
        ),  # int(self.globals.get('DEPTH', FILL_VALUE)),
    )

    set_original_header(nc_file, ds)
    nc_file.WOCE_CTD_FLAG_DESCRIPTION = woce.CTD_FLAG_DESCRIPTION

    create_and_fill_data_variables(nc_file, ds)

    try:
        nobs_data = ds["CTDNOBS"].to_numpy()[0]
        var_number = nc_file.createVariable("number_observations", "i4", ("pressure",))
        var_number.long_name = "number_observations"
        var_number.units = "integer"
        var_number.data_min = np.min(nobs_data)
        var_number.data_max = np.max(nobs_data)
        var_number.C_format = "%1d"
        var_number[:] = nobs_data
    except KeyError:
        pass

    ctd_create_common_variables(nc_file, ds)

    data = bytes(nc_file.close())
    return data
