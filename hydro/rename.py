"""
"""
from importlib.resources import open_text
from csv import DictReader
from typing import List, Mapping, Optional

from xarray import Dataset

import hydro.data


def is_not_none(obj):
    return obj is not None


def rename_with_bookkeeping(
    xarray_obj: Dataset,
    name_dict: Optional[Mapping] = None,
    attrs: Optional[List[str]] = None,
) -> Dataset:
    """
    """

    # lets just noop this case
    if name_dict is None:
        return xarray_obj

    # easy part
    renamed = xarray_obj.rename(name_dict)

    # no bookkeeping to do
    if attrs is None:
        return renamed

    for attr in attrs:
        for var, ds in renamed.filter_by_attrs(**{attr: is_not_none}).items():
            attr_value = ds.attrs[attr]
            for key, value in name_dict.items():
                attr_value = attr_value.replace(key, value)

            ds.attrs[attr] = attr_value

    return renamed


def to_argo_variable_names(xarray_obj: Dataset) -> Dataset:
    with open_text(
        hydro.data, "argo-parameters-list-code-and-b.csv", encoding="utf-8-sig"
    ) as f:
        argo_params = {d["parameter name"]: d for d in DictReader(f)}
