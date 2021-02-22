"""
"""
from typing import List, Mapping, Optional

from xarray import Dataset


def is_not_none(obj):
    return obj is not None


def rename_with_bookkeeping(
    xarray_obj: Dataset,
    name_dict: Optional[Mapping] = None,
    attrs: Optional[List[str]] = None,
) -> Dataset:
    """Find and update all instances of a given variable to a new name.

    Parameters can be referenced in the attributes of separate parameter (e.g. ``ancillary_variables``) and need to be updated appropriately when renaming variables.

    :param xarray.Dataset xarray_obj: A Dataset containing variables, flags, etc.
    :param typing.Mapping name_dict: Mapping of old variable names to new.
    :param typing.List[str] attrs: Names of variable attributes to search through.
    :rtype: xarray.Dataset
    """

    # lets just noop this case
    if name_dict is None:
        return xarray_obj

    # easy part
    renamed = xarray_obj.rename(name_dict)

    # no bookkeeping to do
    if attrs is None:
        return renamed

    # find and search through all variables which have attr
    for attr in attrs:
        for var, ds in renamed.filter_by_attrs(**{attr: is_not_none}).items():
            attr_values = ds.attrs[attr].split(" ")
            for key, value in name_dict.items():
                for idx, attr_value in enumerate(attr_values):
                    if key == attr_value:
                        attr_values[idx] = attr_value.replace(key, value)
            renamed[var].attrs[attr] = " ".join(attr_values)

    return renamed


def to_argo_variable_names(xarray_obj: Dataset) -> Dataset:
    ...
