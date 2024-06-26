from collections import defaultdict

import xarray as xr


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
