from datetime import datetime, timedelta
from warnings import warn

import numpy as np
import numpy.typing as npt
import xarray as xr

from cchdo.hydro.exchange.exceptions import ExchangeError
from cchdo.hydro.types import WHPNameAttr
from cchdo.params import WHPName, WHPNames

DATE = WHPNames["DATE"]
TIME = WHPNames["TIME"]


def _combine_dt_ndarray(
    date_arr: npt.NDArray[np.str_],
    time_arr: npt.NDArray[np.str_] | None = None,
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

    time_arr = time_arr.astype("U4")

    if time_pad:
        if np.any(np.char.str_len(time_arr[time_arr != ""]) < 4):
            warn("Time values are being padded with zeros", stacklevel=2)
        if not np.all(time_arr == ""):
            time_arr[time_arr != ""] = np.char.zfill(time_arr[time_arr != ""], 4)

    arr = np.char.add(np.char.add(date_arr, "T"), time_arr)
    return parse_datetime(arr).astype("datetime64[m]")


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
    date = dataset[date_name.full_nc_name]
    time: xr.DataArray | None = dataset.get(
        time_name.full_nc_name
    )  # not be present, this is allowed

    whp_name: WHPNameAttr = [date_name.full_whp_name, time_name.full_whp_name]
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
        dt_arr.astype("datetime64[ns]"),
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
