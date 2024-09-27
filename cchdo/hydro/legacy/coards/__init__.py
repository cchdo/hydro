"""Legacy COARDS netcdf make from libcchdo ported to take a CCHDO CF/netCDF xarray.Dataset object as input.

The goal is, as much as possible, to use the old code with minimal changes such that the following outputs are identical:

* Exchange -> CF/netCDF -> COARDS netCDF (this library)
* Exchange -> COARDS netCDF (using libcchdo)

The entrypoint function is :func:`to_coards`
"""

import datetime
from csv import DictReader

# TODO: switch to files().joinpath().open when python 3.8 is dropped
# 2023-04-16
from importlib.resources import open_text
from io import BytesIO
from logging import getLogger
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import xarray as xr
from cftime import date2num

from cchdo.hydro import accessors as acc
from cchdo.hydro.legacy import woce

log = getLogger(__name__)
"""logger object for message logging"""

# Load the WHP to nc name mapping
# this was dumped from old libcchdo
with open_text("cchdo.hydro.legacy.coards", "name_netcdf.csv") as params:
    PARAMS = {}
    """mapping of whp names to nc names

    This is loaded at module import time from a dump from the old internal params sqlite database
    """
    for param in DictReader(params):
        PARAMS[f"{param['name']} [{param['mnemonic']}]"] = param

# collection exts
CTD_ZIP_FILE_EXTENSION = "nc_ctd.zip"
"""Filename extention for a zipped collection ctd coards netcdf files"""
BOTTLE_ZIP_FILE_EXTENSION = "nc_hyd.zip"
"""Filename extention for a zipped collection bottle coards netcdf files"""

# These consts are taken directly from libcchdo
FILL_VALUE = -999.0
"""Const from old libcchdo, -999.0"""

QC_SUFFIX = "_QC"
"""Variable name suffix for flag variables"""
FILE_EXTENSION = "nc"
"""filenmae extention for all netcdf files"""

EPOCH = datetime.datetime(1980, 1, 1, 0, 0, 0)
"""dateime referenced in the units of time variables in netCDF files: 1980-01-01"""

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
"""List of WHP names that are ignored when calling :func:`create_and_fill_data_variables`"""

NON_FLOAT_PARAMETERS = ("CTDNOBS",)
"""params not in :attr:`STATIC_PARAMETERS_PER_CAST` that are also ignored by :func:`create_and_fill_data_variables`"""

UNKNOWN = "UNKNOWN"
"""Value used when some string value isn't found

This is mmostly mitigated by the guarantees of the new CF format, but e.g. section id might be missing
"""

UNSPECIFIED_UNITS = "unspecified"
"""Value used when there are no units"""

STRLEN = 40
"""length of char array variables, hardcoded to 40"""


# new behavior will always have the input be a datetime.datetime (Or np.datetime64)
# this function needs modification
def strftime_woce_date_time(dt: xr.DataArray):
    """Take an xr.DataArray with time values in it and convert to strings."""
    if dt is None:
        return (None, None)
    if dt.attrs.get("resolution", 0) >= 1:
        return (dt.dt.strftime("%Y%m%d").item(), None)
    return (dt.dt.strftime("%Y%m%d").item(), dt.dt.strftime("%H%M").item())


# utility functions from libcchdo.formats.netcdf


# name change ascii -> _ascii to avoid builtin conflict
def _ascii(x: str) -> str:
    """Force all codepoints into valid ascii range.

    Works by encoding the str into ascii bytes with the replace err param, then decoding the bytes to str again

    :param x: string with any unicode codepoint in it
    :returns: string with all non ascii codepoints replaced with whatever "replace" does in :py:meth:`str.encode`
    """
    return x.encode("ascii", "replace").decode("ascii")


def simplest_str(s) -> str:
    """Give the simplest string representation.

    If a float is almost equivalent to an integer, swap out for the integer.
    """
    # if type(s) is float:
    if isinstance(s, float):
        # if fns.equal_with_epsilon(s, int(s)):
        # replace with equivalent numpy call
        if np.isclose(s, int(s), atol=1e-6):
            s = int(s)
    return str(s)


def _pad_station_cast(x: str) -> str:
    """Pad a station or cast identifier out to 5 characters.

    This is usually for use in a file name.

    :param x: a string to be padded
    :type x: str
    """
    return simplest_str(x).rjust(5, "0")


def get_filename(expocode, station, cast, extension):
    """Generate the filename for COARDS netCDF files.

    Was ported directly from libcchdo and should have the same formatting behavior
    """
    if extension not in ["hy1", "ctd"]:
        log.warning("File extension is not recognized.")
    station = _pad_station_cast(station)
    cast = _pad_station_cast(cast)

    stem = "_".join((expocode, station, cast, extension))
    return f"{stem}.{FILE_EXTENSION}"


def minutes_since_epoch(dt: xr.DataArray, epoch, error=-9):
    """Make the time value for netCDF files.

    The custom implimentation in libcchdo was discarded in favor of the date2num function from cftime.
    Not sure if cftime exsited in the netCDF4 python library at the time.
    """
    return date2num(
        dt.values.astype("<M8[ms]").astype(datetime.datetime),
        epoch,
        calendar="proleptic_gregorian",
    )


# end utility


def get_coards_global_attributes(ds: xr.Dataset, *, profile_type: Literal["B", "C"]):
    """Makes the global attributes of a WHP COARDS netCDF File.

    The order of the attributes is important/fixed, same with case"""

    data_types = {
        "B": "WOCE Bottle",
        "C": "WOCE CTD",
    }

    attrs = {
        "EXPOCODE": ds.expocode.item(),
        "Conventions": "COARDS/WOCE",
        "WOCE_VERSION": "3.0",
    }

    if woce_id := ds.get("section_id"):
        attrs["WOCE_ID"] = woce_id.item()

    attrs["DATA_TYPE"] = data_types[profile_type]
    attrs["STATION_NUMBER"] = ds.station.item()
    attrs["CAST_NUMBER"] = ds["cast"].astype(str).item()

    if bottom_depth := ds.get("btm_depth"):
        attrs["BOTTOM_DEPTH_METERS"] = int(bottom_depth.fillna(FILL_VALUE).item())
    else:
        attrs["BOTTOM_DEPTH_METERS"] = FILL_VALUE

    attrs["Creation_Time"] = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    _comments = ds.attrs.get("comments", "").splitlines()
    if len(_comments) == 0:
        _comments = [""]
    og_header = "\n".join([_comments[0], *[f"#{line}" for line in _comments[1:]], ""])
    attrs["ORIGINAL_HEADER"] = og_header

    if profile_type == "B":
        ds = ds.stack(ex=("N_PROF", "N_LEVELS"))
        ds = ds.isel(ex=(ds.sample != ""))
        bottle_column = ds.get("bottle_number", ds["sample"])
        attrs["BOTTLE_NUMBERS"] = " ".join(map(simplest_str, bottle_column.values))

        if (btl_quality_codes := ds.get("bottle_number_qc")) is not None:
            attrs["BOTTLE_QUALITY_CODES"] = btl_quality_codes.to_numpy().astype(
                np.int16
            )

        attrs["WOCE_BOTTLE_FLAG_DESCRIPTION"] = woce.BOTTLE_FLAG_DESCRIPTION
        attrs["WOCE_WATER_SAMPLE_FLAG_DESCRIPTION"] = woce.WATER_SAMPLE_FLAG_DESCRIPTION

    if profile_type == "C":
        attrs["WOCE_CTD_FLAG_DESCRIPTION"] = woce.CTD_FLAG_DESCRIPTION

    return attrs


def get_dataarrays(ds: xr.Dataset):
    dataarrays = {}
    for whpname, variable in ds.cchdo.to_whp_columns(compact=True).items():
        attrs = {}

        parameter_name = whpname.whp_name

        parameter_unit = whpname.whp_unit or ""
        parameter_key = f"{parameter_name} [{parameter_unit}]"

        # Hacks to match behavior of previous
        if parameter_name == "INSTRUMENT_ID":
            continue

        if parameter_name in STATIC_PARAMETERS_PER_CAST:
            continue

        if parameter_name in NON_FLOAT_PARAMETERS:
            continue

        # porting note: this logic is to match the libcchdo COARDS netcdf varnames
        parameter = PARAMS.get(parameter_key, {})
        _pname = parameter.get("name_netcdf")
        if not _pname:
            log.debug(
                "No netcdf name for %s. Using mnemonic %s.",
                parameter_key,
                parameter_name,
            )
            _pname = parameter_name
        if not _pname:
            raise AttributeError(f"No name found for {parameter_name}")
        pname = _ascii(_pname)

        # porting note: this was taken from the orig libcchdo
        # XXX HACK
        if pname == "oxygen1":
            pname = "oxygen"

        attrs["long_name"] = pname
        if pname == "pressure":
            attrs["positive"] = "down"

        units = UNSPECIFIED_UNITS
        if parameter.get("units_name"):
            units = parameter["units_name"]
        elif parameter_unit != "":
            units = parameter_unit
        attrs["units"] = _ascii(units)

        if parameter_name == "BTL_DATE":
            data = variable.dt.strftime("%Y%m%d").astype(float).to_numpy()
        elif parameter_name == "BTL_TIME":
            data = variable.dt.strftime("%H%M").astype(float).to_numpy()
        elif parameter_name == "CTDETIME":
            nat_mask = np.isnat(variable)
            data = variable.values.astype("timedelta64[s]").astype("float64")
            data[nat_mask] = np.nan
        else:
            data = variable.to_numpy()

        # porting note: hack for new string params that would crash the old converter anyway
        if data.dtype.kind in "OSU":
            data = np.full_like(data, np.nan, dtype=float)

        if data.dtype.kind in "iuf":
            if np.all(np.isnan(data)):
                attrs["data_min"] = float("-inf")  # type: ignore
                attrs["data_max"] = float("inf")  # type: ignore
            else:
                attrs["data_min"] = np.nanmin(data)
                attrs["data_max"] = np.nanmax(data)

        if parameter.get("format"):
            attrs["C_format"] = _ascii(parameter["format"])
        else:
            # TODO TEST this
            log.debug("Parameter %s has no format. defaulting to '%%f'", parameter_name)
            attrs["C_format"] = "%f"
        if attrs["C_format"].endswith("s"):
            log.debug(
                "Parameter %s does not have a format string acceptable for "
                "numeric data. Defaulting to '%%f' to prevent ncdump "
                "segfault.",
                parameter_name,
            )
            attrs["C_format"] = "%f"

        attrs["WHPO_Variable_Name"] = parameter_name

        dataarrays[pname] = xr.DataArray(
            data, dims=["pressure"], name=pname, attrs=attrs
        )
        dataarrays[pname].encoding["_FillValue"] = None

        if (qc_variable := variable.attrs.get(acc.FLAG_NAME)) is not None:
            qc_name = pname + QC_SUFFIX
            dataarrays[pname].attrs["OBS_QC_VARIABLE"] = qc_name

            qc_attrs = {
                "long_name": qc_name + "_flag",
                "units": "woce_flags",
                "C_format": "%1d",
            }
            qc_data = np.nan_to_num(qc_variable.to_numpy(), nan=9).astype("int16")
            qc_dataarray = xr.DataArray(
                qc_data, dims=["pressure"], name=qc_name, attrs=qc_attrs
            )
            dataarrays[qc_name] = qc_dataarray

    return dataarrays


def get_common_variables(ds: xr.Dataset):
    # In the origional it creates these all manually (not in a loop)
    common_variables = {}
    time_units = f"minutes since {EPOCH.isoformat(' ')}"
    time_data = int(minutes_since_epoch(ds.time, time_units).item())
    common_variables["time"] = xr.DataArray(
        [time_data],
        dims=("time"),
        attrs={
            "long_name": "time",
            "units": time_units,
            "data_min": time_data,
            "data_max": time_data,
            "C_format": "%10d",
        },
    )

    latitude = float(ds.latitude.item())
    common_variables["latitude"] = xr.DataArray(
        np.array([latitude], dtype="float32"),
        dims=["latitude"],
        attrs={
            "long_name": "latitude",
            "units": "degrees_N",
            "data_min": latitude,
            "data_max": latitude,
            "C_format": "%9.4f",
        },
    )
    common_variables["latitude"].encoding["_FillValue"] = None

    longitude = float(ds.longitude.item())
    common_variables["longitude"] = xr.DataArray(
        np.array([longitude], dtype="float32"),
        dims=["longitude"],
        attrs={
            "long_name": "longitude",
            "units": "degrees_E",
            "data_min": longitude,
            "data_max": longitude,
            "C_format": "%9.4f",
        },
    )
    common_variables["longitude"].encoding["_FillValue"] = None

    strs_woce_datetime = strftime_woce_date_time(ds.time)

    date = int(strs_woce_datetime[0] or -9)
    common_variables["woce_date"] = xr.DataArray(
        [date],
        dims=["time"],
        attrs={
            "long_name": "WOCE date",
            "units": "yyyymmdd UTC",
            "data_min": date,
            "data_max": date,
            "C_format": "%8d",
        },
    )

    if strs_woce_datetime[1]:
        time = int(strs_woce_datetime[1] or -9)
        common_variables["woce_time"] = xr.DataArray(
            np.array([time], dtype="int16"),
            dims=["time"],
            attrs={
                "long_name": "WOCE time",
                "units": "hhmm UTC",
                "data_min": time,
                "data_max": time,
                "C_format": "%4d",
            },
        )

    station = simplest_str(ds.station.item()).ljust(STRLEN)[:STRLEN]
    common_variables["station"] = xr.DataArray(
        station,
        attrs={"long_name": "STATION", "units": UNSPECIFIED_UNITS, "C_format": "%s"},
    )
    common_variables["station"].encoding["char_dim_name"] = "string_dimension"

    station = simplest_str(ds.cast.item()).ljust(STRLEN)[:STRLEN]
    common_variables["cast"] = xr.DataArray(
        station,
        attrs={"long_name": "CAST", "units": UNSPECIFIED_UNITS, "C_format": "%s"},
    )
    common_variables["cast"].encoding["char_dim_name"] = "string_dimension"

    return common_variables


def write_bottle(ds: xr.Dataset) -> bytes:
    attrs = get_coards_global_attributes(ds, profile_type="B")
    data_vars = get_dataarrays(ds)
    common_vars = get_common_variables(ds)

    nc_file = xr.Dataset(data_vars={**data_vars, **common_vars}, attrs=attrs)
    return nc_file.to_netcdf(format="NETCDF3_CLASSIC")


def write_ctd(ds: xr.Dataset) -> bytes:
    attrs = get_coards_global_attributes(ds, profile_type="C")
    data_vars = get_dataarrays(ds)

    if (ctd_nobs := ds.get("ctd_number_of_observations")) is not None:
        stacked_ctd_nobs = ctd_nobs.stack(ex=("N_PROF", "N_LEVELS"))
        valid_ctd_nobs = stacked_ctd_nobs.isel(ex=(stacked_ctd_nobs.sample != ""))
        nobs_data = np.nan_to_num(valid_ctd_nobs.to_numpy(), nan=-999).astype("int64")
        data_vars["number_observations"] = xr.DataArray(
            nobs_data,
            dims=["pressure"],
            attrs={
                "long_name": "number_observations",
                "units": "integer",
                "data_min": np.min(nobs_data),
                "data_max": np.max(nobs_data),
                "C_format": "%1d",
            },
        )
        data_vars["number_observations"].encoding["_FillValue"] = -999
    common_vars = get_common_variables(ds)

    nc_file = xr.Dataset(data_vars={**data_vars, **common_vars}, attrs=attrs)
    return nc_file.to_netcdf(format="NETCDF3_CLASSIC")


def to_coards(ds: xr.Dataset) -> bytes:
    """Convert an xr.Dataset to a zipfile with COARDS netCDF files inside.

    This function does support mixed CTD and Bottle datasets and will convert using profile_type var on a per profile basis.


    :param ds: A dataset conforming to CCHDO CF/netCDF
    :returns: a zipfile with one or more COARDS netCDF files as members.
    """
    output_files = {}
    for _, profile in ds.groupby("N_PROF", squeeze=False):
        if profile.profile_type.item() == "C":
            data = write_ctd(profile)
            extension = "ctd"
        elif profile.profile_type.item() == "B":
            extension = "hy1"
            data = write_bottle(profile)
        else:
            raise NotImplementedError()

        filename = get_filename(
            profile.expocode.item(),
            profile.station.item(),
            profile.cast.item(),
            extension=extension,
        )
        output_files[filename] = data

    output_zip = BytesIO()
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zipfile:
        for fname, data in output_files.items():
            zipfile.writestr(fname, data)

    output_zip.seek(0)
    return output_zip.read()
