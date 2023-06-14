import xarray as xr
import numpy as np

from ..core import create_new, add_profile


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

    result = create_new()

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

    base = create_new()

    result = add_profile(base, "318M", "test", 1, "2020-01-01T00:00:00", 0, 0, "B")

    xr.testing.assert_identical(result, expected)
