from collections import defaultdict

import numpy as np
import xarray as xr

from cchdo.hydro.exchange.exceptions import ExchangeDataFlagPairError
from cchdo.hydro.flags import (
    ExchangeBottleFlag,
    ExchangeCTDFlag,
    ExchangeSampleFlag,
)
from cchdo.hydro.sorting import sort_ds


def check_sorted(dataset: xr.Dataset) -> bool:
    """Check that the dataset is sorted by the rules in :func:`sort_ds`"""
    sorted_ds = sort_ds(dataset.copy(deep=True))

    return all(
        [
            np.allclose(sorted_ds.pressure, dataset.pressure, equal_nan=True),
            np.all(
                (sorted_ds.time == dataset.time)
                | (np.isnat(sorted_ds.time) == np.isnat(dataset.time))
            ),
            np.allclose(sorted_ds.latitude, dataset.latitude, equal_nan=True),
            np.allclose(sorted_ds.longitude, dataset.longitude, equal_nan=True),
        ]
    )


def check_ancillary_variables(ds: xr.Dataset):
    """Check that everything in an ancillary_variables attribute appears as a variable
    Check that every variable that is known ancillary appears in at least one ancillary_variable attribute
    """
    looks_ancillary_suffixes = ("_qc", "_error")

    ancillary_variables_attrs = defaultdict(list)
    looks_ancillary = set()

    for name, variable in ds.variables.items():
        if not isinstance(name, str):
            raise ValueError(f"variable names must be strings not {name}")

        if any(name.endswith(suffix) for suffix in looks_ancillary_suffixes):
            looks_ancillary.add(name)

        if variable.attrs.get("ancillary_variables") is None:
            continue

        for ancillary in variable.attrs["ancillary_variables"].split():
            ancillary_variables_attrs[ancillary].append(name)

    if errors := ancillary_variables_attrs.keys() - ds.variables.keys():
        raise ValueError(errors)

    if errors := looks_ancillary - ancillary_variables_attrs.keys():
        raise ValueError(errors)


def check_flags(dataset: xr.Dataset, raises=True):
    """Check WOCE flag values agaisnt their param and ensure that the param either has a value or is "nan" depedning on the flag definition.

    Return a boolean array of invalid locations?
    """
    woce_flags = {
        "WOCESAMPLE": ExchangeBottleFlag,
        "WOCECTD": ExchangeCTDFlag,
        "WOCEBOTTLE": ExchangeSampleFlag,
    }
    flag_has_value = {
        "WOCESAMPLE": {flag.value: flag.has_value for flag in ExchangeBottleFlag},
        "WOCECTD": {flag.value: flag.has_value for flag in ExchangeCTDFlag},
        "WOCEBOTTLE": {flag.value: flag.has_value for flag in ExchangeSampleFlag},
    }
    # In some cases, a coordinate variable might have flags, so we are not using filter_by_attrs
    # get all the flag vars (that also have conventions)
    flag_vars = []
    for var_name in dataset.variables:
        # do not replace the above with .items() it will give you xr.Variable objects that you don't want to use
        # the following gets a real xr.DataArray
        data = dataset[var_name]
        if not {"standard_name", "conventions"} <= data.attrs.keys():
            continue
        if not any(flag in data.attrs["conventions"] for flag in woce_flags):
            continue
        if "status_flag" in data.attrs["standard_name"]:
            flag_vars.append(var_name)

    # match flags with their data vars
    # it is legal in CF for one set of flags to apply to multiple vars
    flag_errors = {}
    for flag_var in flag_vars:
        # get the flag and check attrs for defs
        flag_da = dataset[flag_var]
        conventions = None
        for flag in woce_flags:
            if flag_da.attrs.get("conventions", "").startswith(flag):
                conventions = flag
                break

        # we don't know these flags, skip the check
        if not conventions:
            continue

        allowed_values = np.array(list(flag_has_value[conventions]))
        illegal_flags = ~flag_da.fillna(9).isin(allowed_values)
        if np.any(illegal_flags):
            illegal_flags.attrs["comments"] = (
                f"This is a boolean array in the same shape as '{flag_da.name}' which is truthy where invalid values exist"
            )
            flag_errors[f"{flag_da.name}_value_errors"] = illegal_flags
            continue

        for var_name in dataset.variables:
            data = dataset[var_name]
            if "ancillary_variables" not in data.attrs:
                continue
            if flag_var not in data.attrs["ancillary_variables"].split(" "):
                continue

            # check data against flags
            has_fill_f = [
                flag
                for flag, value in flag_has_value[conventions].items()
                if value is False
            ]

            has_fill = flag_da.isin(has_fill_f) | np.isnan(flag_da)

            # TODO deal with strs

            if np.issubdtype(data.values.dtype, np.number):
                fill_value_mismatch: xr.DataArray = ~(np.isfinite(data) ^ has_fill)  # type: ignore[assignment]
                if np.any(fill_value_mismatch):
                    fill_value_mismatch.attrs["comments"] = (
                        f"This is a boolean array in the same shape as '{data.name}' which is truthy where invalid values exist"
                    )
                    flag_errors[f"{data.name}_value_errors"] = fill_value_mismatch

    flag_errors_ds = xr.Dataset(flag_errors)
    if raises and any(flag_errors_ds):
        raise ExchangeDataFlagPairError(flag_errors_ds)

    return flag_errors_ds
