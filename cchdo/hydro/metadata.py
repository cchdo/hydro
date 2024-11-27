from collections import defaultdict
from itertools import groupby, islice
from logging import getLogger

import xarray as xr

log = getLogger(__name__)


def all_equal(iterable, key=None):
    "Returns True if all the elements are equal to each other."
    # see https://docs.python.org/3/library/itertools.html#itertools-recipes
    return len(list(islice(groupby(iterable, key), 2))) <= 1


SAME_KEYS = (
    "processing_level",
    "comment",
    "creator_name",
)
ALLOWED_DIFFER = (
    "date_modified",
    "date_metadata_modified",
)


def validate(ds: xr.Dataset):
    exceptions = []

    # cannot use filter_by_attrs since we want to filter on the coordinates too
    projects = defaultdict(list)
    for name, da in ds.variables.items():
        if (project := da.attrs.get("project")) is not None:
            projects[project].append(name)

    for project, vars in projects.items():
        log.debug(f"Checking project '{project}' which includes {vars}")
        for key in SAME_KEYS:
            values = {var: ds[var].attrs.get(key) for var in vars}
            valid = all_equal(values.values())
            if not valid:
                exception = ValueError(
                    f"Project '{project}' key '{key}' is not the same: {values}"
                )
                log.debug(exception)
                exceptions.append(exception)
    if exceptions:
        raise ExceptionGroup("Metadata has failed to validate", exceptions)
