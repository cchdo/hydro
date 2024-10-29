import io

import numpy as np
import pytest
import xarray as xr

from cchdo.hydro import core
from cchdo.hydro.exchange import read_exchange
from cchdo.hydro.exchange.helpers import simple_bottle_exchange
from cchdo.params import WHPNames


def test_create_new():
    expected = xr.Dataset(
        {
            "profile_type": xr.DataArray(
                np.empty((0), np.dtype("U")),
                dims=["N_PROF"],
                attrs={},
                name="profile_type",
            ),
            "geometry_container": xr.DataArray(
                np.array(np.nan, np.dtype("float64")),
                dims=[],
                attrs={
                    "geometry_type": "point",
                    "node_coordinates": "longitude latitude",
                },
            ),
        },
        coords={
            "expocode": xr.DataArray(
                np.empty((0), np.dtype("U")),
                dims=["N_PROF"],
                attrs={"whp_name": "EXPOCODE", "geometry": "geometry_container"},
            ),
            "station": xr.DataArray(
                np.empty((0), np.dtype("U")),
                dims=["N_PROF"],
                attrs={"whp_name": "STNNBR", "geometry": "geometry_container"},
            ),
            "cast": xr.DataArray(
                np.empty((0), np.dtype("int32")),
                dims=["N_PROF"],
                attrs={"whp_name": "CASTNO", "geometry": "geometry_container"},
            ),
            "sample": xr.DataArray(
                np.empty((0, 0), np.dtype("U")),
                dims=["N_PROF", "N_LEVELS"],
                attrs={"whp_name": "SAMPNO"},
            ),
            "time": xr.DataArray(
                np.empty((0), np.dtype("datetime64[ns]")),
                dims=["N_PROF"],
                attrs={
                    "standard_name": "time",
                    "whp_name": "DATE",
                    "resolution": 1,
                    "axis": "T",
                    "geometry": "geometry_container",
                },
            ),
            "latitude": xr.DataArray(
                np.empty((0), np.dtype("float64")),
                dims=["N_PROF"],
                attrs={
                    "standard_name": "latitude",
                    "units": "degree_north",
                    "C_format": "%9.4f",
                    "C_format_source": "database",
                    "axis": "Y",
                    "whp_name": "LATITUDE",
                },
            ),
            "longitude": xr.DataArray(
                np.empty((0), np.dtype("float64")),
                dims=["N_PROF"],
                attrs={
                    "standard_name": "longitude",
                    "units": "degree_east",
                    "C_format": "%9.4f",
                    "C_format_source": "database",
                    "axis": "X",
                    "whp_name": "LONGITUDE",
                },
            ),
            "pressure": xr.DataArray(
                np.empty((0, 0), np.dtype("float64")),
                dims=["N_PROF", "N_LEVELS"],
                attrs={
                    "standard_name": "sea_water_pressure",
                    "units": "dbar",
                    "C_format": "%9.1f",
                    "C_format_source": "database",
                    "axis": "Z",
                    "whp_name": "CTDPRS",
                    "whp_unit": "DBAR",
                    "positive": "down",
                },
            ),
        },
    )

    result = core.create_new()

    xr.testing.assert_identical(result, expected)


def test_add_profile():
    expected = xr.Dataset(
        {
            "profile_type": xr.DataArray(
                np.array(["B"], np.dtype("U")),
                dims=["N_PROF"],
                attrs={},
                name="profile_type",
            ),
            "geometry_container": xr.DataArray(
                np.array(np.nan, np.dtype("float64")),
                dims=[],
                attrs={
                    "geometry_type": "point",
                    "node_coordinates": "longitude latitude",
                },
            ),
        },
        coords={
            "expocode": xr.DataArray(
                np.array(["318M"], np.dtype("U")),
                dims=["N_PROF"],
                attrs={"whp_name": "EXPOCODE", "geometry": "geometry_container"},
            ),
            "station": xr.DataArray(
                np.array(["test"], np.dtype("U")),
                dims=["N_PROF"],
                attrs={"whp_name": "STNNBR", "geometry": "geometry_container"},
            ),
            "cast": xr.DataArray(
                np.array([1], np.dtype("int32")),
                dims=["N_PROF"],
                attrs={"whp_name": "CASTNO", "geometry": "geometry_container"},
            ),
            "sample": xr.DataArray(
                np.empty((1, 0), np.dtype("U")),
                dims=["N_PROF", "N_LEVELS"],
                attrs={"whp_name": "SAMPNO"},
            ),
            "time": xr.DataArray(
                np.array(["2020-01-01T00:00:00"], np.dtype("datetime64[ns]")),
                dims=["N_PROF"],
                attrs={
                    "standard_name": "time",
                    "whp_name": "DATE",
                    "resolution": 1,
                    "axis": "T",
                    "geometry": "geometry_container",
                },
            ),
            "latitude": xr.DataArray(
                np.array([0], np.dtype("float64")),
                dims=["N_PROF"],
                attrs={
                    "standard_name": "latitude",
                    "units": "degree_north",
                    "C_format": "%9.4f",
                    "C_format_source": "database",
                    "axis": "Y",
                    "whp_name": "LATITUDE",
                },
            ),
            "longitude": xr.DataArray(
                np.array([0], np.dtype("float64")),
                dims=["N_PROF"],
                attrs={
                    "standard_name": "longitude",
                    "units": "degree_east",
                    "C_format": "%9.4f",
                    "C_format_source": "database",
                    "axis": "X",
                    "whp_name": "LONGITUDE",
                },
            ),
            "pressure": xr.DataArray(
                np.empty((1, 0), np.dtype("float64")),
                dims=["N_PROF", "N_LEVELS"],
                attrs={
                    "standard_name": "sea_water_pressure",
                    "units": "dbar",
                    "C_format": "%9.1f",
                    "C_format_source": "database",
                    "axis": "Z",
                    "whp_name": "CTDPRS",
                    "whp_unit": "DBAR",
                    "positive": "down",
                },
            ),
        },
    )

    base = core.create_new()

    result = core.add_profile(base, "318M", "test", 1, "2020-01-01T00:00:00", 0, 0, "B")

    xr.testing.assert_identical(result, expected)


def test_add_param():
    # The easiest way I could think of to test this was to make some exchange inputs
    # one with and without some parameter and try use the functions to test the addition
    # and removal of a param
    # TODO: parameterize this to test the parameter space
    params = ("DELC14", "DELC14_FLAG_W", "C14ERR")
    units = ("/MILLE", "", "/MILLE")
    data = ("-999", "9", "-999")
    ds = read_exchange(
        io.BytesIO(simple_bottle_exchange()), precision_source="database"
    )
    ds_param = read_exchange(
        io.BytesIO(
            simple_bottle_exchange(params=params[:1], units=units[:1], data=data[:1])
        ),
        precision_source="database",
    )
    ds_param_flag = read_exchange(
        io.BytesIO(
            simple_bottle_exchange(params=params[:2], units=units[:2], data=data[:2])
        ),
        precision_source="database",
    )
    ds_param_flag_error = read_exchange(
        io.BytesIO(
            simple_bottle_exchange(params=params[:3], units=units[:3], data=data[:3])
        ),
        precision_source="database",
    )

    testing_ds_param = core.add_param(ds, WHPNames["DELC14 [/MILLE]"])
    xr.testing.assert_identical(ds_param, testing_ds_param)
    testing_ds_param_flag = core.add_param(
        ds, WHPNames["DELC14 [/MILLE]"], with_flag=True
    )
    xr.testing.assert_identical(ds_param_flag, testing_ds_param_flag)
    testing_ds_param_flag_error = core.add_param(
        ds, WHPNames["DELC14 [/MILLE]"], with_flag=True, with_error=True
    )
    xr.testing.assert_identical(ds_param_flag_error, testing_ds_param_flag_error)


@pytest.mark.parametrize("require_empty", [True, False])
@pytest.mark.parametrize("empty", [True, False])
@pytest.mark.parametrize("error", [True, False])
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("param", [True, False])
def test_remove_param(param, flag, error, empty, require_empty):
    # TODO see above todo

    if param is True:
        flag, error = True, True

    bools = (param, flag, error)
    params = ("DELC14", "DELC14_FLAG_W", "C14ERR")
    units = ("/MILLE", "", "/MILLE")
    if empty:
        data = ("-999", "9", "-999")
    else:
        data = ("0.1", "2", "0.1")

    if not all(bools):
        _, _params, _units, _data = zip(
            *filter(lambda x: not x[0], zip(bools, params, units, data))
        )
    else:
        _params = tuple()
        _units = tuple()
        _data = tuple()

    expected_ds = read_exchange(
        io.BytesIO(simple_bottle_exchange(params=_params, units=_units, data=_data)),
        precision_source="database",
    )
    input_ds = read_exchange(
        io.BytesIO(simple_bottle_exchange(params=params, units=units, data=data)),
        precision_source="database",
    )

    if empty is False and require_empty is True and any(bools):
        with pytest.raises(ValueError):
            testing_ds = core.remove_param(
                input_ds,
                "DELC14 [/MILLE]",
                delete_param=param,
                delete_flag=flag,
                delete_error=error,
                require_empty=require_empty,
            )
    else:
        testing_ds = core.remove_param(
            input_ds,
            "DELC14 [/MILLE]",
            delete_param=param,
            delete_flag=flag,
            delete_error=error,
            require_empty=require_empty,
        )
        xr.testing.assert_identical(expected_ds, testing_ds)


def test_add_param_cdom():
    params = (
        "CDOM300",
        "CDOM325",
    )
    units = (
        "/METER",
        "/METER",
    )
    data = (
        "-999",
        "-999",
    )
    ds = read_exchange(
        io.BytesIO(simple_bottle_exchange()), precision_source="database"
    )
    ds_param_300 = read_exchange(
        io.BytesIO(
            simple_bottle_exchange(params=params[:1], units=units[:1], data=data[:1])
        ),
        precision_source="database",
    )
    ds_param_325 = read_exchange(
        io.BytesIO(
            simple_bottle_exchange(params=params[1:], units=units[1:], data=data[1:])
        ),
        precision_source="database",
    )
    ds_param_cdom = read_exchange(
        io.BytesIO(simple_bottle_exchange(params=params, units=units, data=data)),
        precision_source="database",
    )

    testing_ds_param = core.add_param(ds, WHPNames["CDOM300 [/METER]"])
    xr.testing.assert_identical(ds_param_300, testing_ds_param)

    testing_ds_param = core.add_param(ds_param_300, WHPNames["CDOM325 [/METER]"])
    xr.testing.assert_identical(ds_param_cdom, testing_ds_param)

    testing_ds_param = core.add_param(ds, WHPNames["CDOM325 [/METER]"])
    xr.testing.assert_identical(ds_param_325, testing_ds_param)

    testing_ds_param = core.add_param(ds_param_325, WHPNames["CDOM300 [/METER]"])
    xr.testing.assert_identical(ds_param_cdom, testing_ds_param)
