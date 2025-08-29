from typing import Any, TypeGuard

import numpy as np
import numpy.typing as npt
import xarray as xr

from cchdo.hydro.consts import COORDS, DIMS, GEOMETRY_VARS, TIME
from cchdo.hydro.types import FileType, FileTypeType
from cchdo.params import WHPNames


def _is_all_dataarray(val: list[Any]) -> TypeGuard[list[xr.DataArray]]:
    return all(isinstance(obj, xr.DataArray) for obj in val)


def all_same(ndarr: np.ndarray) -> np.bool_:
    """Test if all the values of an ndarray are the same value"""
    if np.issubdtype(ndarr.dtype, np.number) and np.isnan(ndarr.flat[0]):
        return np.all(np.isnan(ndarr))
    return np.all(ndarr == ndarr.flat[0])


def add_profile_type(dataset: xr.Dataset, ftype: FileTypeType) -> xr.Dataset:
    """Adds a `profile_type` string variable to the dataset.

    This is for ODV compatability

    .. warning::
      Currently mixed profile types are not supported
    """
    ftype = FileType(ftype)

    profile_type = xr.DataArray(
        np.full(dataset.sizes["N_PROF"], fill_value=ftype.value, dtype="U1"),
        name="profile_type",
        dims=DIMS[0],
    )
    profile_type.encoding["dtype"] = "S1"

    dataset["profile_type"] = profile_type
    return dataset


def add_geometry_var(dataset: xr.Dataset) -> xr.Dataset:
    """Adds a CF-1.8 Geometry container variable to the dataset

    This allows for compatabiltiy with tools like gdal
    """
    dataset = dataset.copy()

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


def add_cdom_coordinate(dataset: xr.Dataset) -> xr.Dataset:
    """Find all the paraters in the cdom group and add their wavelength in a new coordinate"""

    cdom_names = list(filter(lambda x: x.nc_group == "cdom", WHPNames.values()))

    cdom_names = sorted(cdom_names, key=lambda x: x.radiation_wavelength or 0)

    # NM this needs to be sorted by wavelength...
    cdom_data = [
        dataset[name.full_nc_name]
        for name in cdom_names
        if name.full_nc_name in dataset
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
    c_formats = []
    for dataarray in cdom_data:
        whp_name = dataarray.attrs["whp_name"]
        whp_unit = dataarray.attrs["whp_unit"]
        whpname = WHPNames[(whp_name, whp_unit)]
        radiation_wavelengths.append(whpname.radiation_wavelength)
        if "C_format" in dataarray.attrs:
            c_formats.append(dataarray.attrs["C_format"])

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

    new_cdom_attrs = {
        **first.attrs,
        "whp_name": "CDOM{CDOM_WAVELENGTHS}",
        "C_format": max(c_formats),
    }

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
            del dataset[old_name.full_nc_name]
        except KeyError:
            pass

    for old_name in cdom_names:
        try:
            del dataset[f"{old_name.full_nc_name}_qc"]
        except KeyError:
            pass

    return dataset


def flatten_cdom_coordinate(dataset: xr.Dataset) -> xr.Dataset:
    """Takes the a dataset with a CDOM wavelength and explocdes it back into individual variables"""
    if "cdom" not in dataset:
        return dataset

    keys = ["cdom", "CDOM_WAVELENGTHS"]
    if "cdom_qc" in dataset:
        keys.append("cdom_qc")

    ds = dataset.copy()
    cdom_var = ds[keys]
    for cdom_wavelength, arr in cdom_var.groupby("CDOM_WAVELENGTHS", squeeze=False):
        arr = arr.squeeze("CDOM_WAVELENGTHS")
        cdom = arr["cdom"].copy()

        cdom_qc = arr.get("cdom_qc")
        if cdom_qc is not None:
            cdom_qc = cdom_qc.copy()

        cdom.attrs["whp_name"] = cdom.attrs["whp_name"].format(
            CDOM_WAVELENGTHS=cdom_wavelength
        )

        whp_name = WHPNames[f"{cdom.attrs['whp_name']} [{cdom.attrs['whp_unit']}]"]

        if cdom_qc is not None:
            cdom.attrs["ancillary_variables"] = whp_name.nc_name_flag

        ds[whp_name.nc_name] = cdom
        if cdom_qc is not None:
            ds[whp_name.nc_name_flag] = cdom_qc

    return ds.drop_vars(keys)


def extract_numeric_precisions(
    data: list[str] | npt.NDArray[np.str_],
) -> npt.NDArray[np.int_]:
    """Get the numeric precision of a printed decimal number"""
    # magic number explain: np.char.partition expands each element into a 3-tuple
    # of (pre, sep, post) of some sep, in our case a "." char.
    # We only want the post bits [idx 2] (the number of chars after a decimal seperator)
    # of the last axis.
    numeric_parts = np.char.partition(data, ".")[..., 2]
    str_lens = np.char.str_len(numeric_parts)
    return np.max(str_lens, axis=0)


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
