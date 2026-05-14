"""Functions for adding ACDD/discoverability attributes to datasets based on their contents"""

import numpy as np
import xarray as xr


def temporal(ds: xr.Dataset) -> xr.Dataset:
    """Set the temporal extent global attributes

    This includes:
    * time_coverage_start
    * time_coverage_end
    """
    ds_ = ds.copy()

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    attrs = {
        "time_coverage_start": ds_.time.min().dt.strftime(fmt).item(),
        "time_coverage_end": ds_.time.max().dt.strftime(fmt).item(),
    }

    ds_.attrs.update(attrs)
    return ds_


def geospatial(ds: xr.Dataset) -> xr.Dataset:
    """Set the geospatial extent global attributes

    These include:
    * geospatial_lat_min
    * geospatial_lat_max
    * geospatial_lon_min
    * geospatial_lon_max
    * geospatial_vertical_min
    * geospatial_vertical_max
    * geospatial_vertical_positive (always "down")
    * geospatial_vertical_units (always "dbar")
    * box

    Note that according to the "spec" of ACDD 1.3 if the lon min is greater
    than the lon max, this indicates that the geospatial extent crosses the
    discontinuity meridian. This function assumes that if the difference
    between the min/max of lon is larger than 300 degrees that crossing is
    occurring.

    The behavior if the ship goes to a longitude singularity such as the north
    pole not explicitly handled.

    The ACDD 1.3 "spec" also allows for the pressure units of bar to be used
    for the vertical units.

    "box" is not defined in ACDD but instead schema.org and is for JSON-LD.
    From the google Dataset documentation:
    > Points inside box, circle, line, or polygon properties must be expressed
    > as a space separated pair of two values corresponding to latitude and
    > longitude (in that order).

    Which is nicely incompatible with JSON-LD and also doesn't use WKT
    """

    ds_ = ds.copy()

    attrs = {
        "geospatial_lat_min": ds_.latitude.min(skipna=True).item(),
        "geospatial_lat_max": ds_.latitude.max(skipna=True).item(),
        "geospatial_lon_min": ds_.longitude.min(skipna=True).item(),
        "geospatial_lon_max": ds_.longitude.max(skipna=True).item(),
        "geospatial_vertical_min": ds_.pressure.min(skipna=True).item(),
        "geospatial_vertical_max": ds_.pressure.max(skipna=True).item(),
        "geospatial_vertical_positive": ds_.pressure.attrs["positive"],
        "geospatial_vertical_units": ds_.pressure.attrs["units"],
    }
    # Handle basic longitude crossing a discontinuity
    if (attrs["geospatial_lon_max"] - attrs["geospatial_lon_min"]) > 300:
        attrs["geospatial_lon_max"] = np.nanmax(
            ds_.longitude.values, where=ds_.longitude.values < 0, initial=-180
        ).item()
        attrs["geospatial_lon_min"] = np.nanmin(
            ds_.longitude.values, where=ds_.longitude.values > 0, initial=180
        ).item()

    attrs["box"] = (
        f"""{attrs["geospatial_lat_min"]} {attrs["geospatial_lon_min"]} {attrs["geospatial_lat_min"]} {attrs["geospatial_lon_max"]}"""
    )

    ds_.attrs.update(attrs)
    return ds_
