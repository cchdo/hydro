import numpy as np
import xarray as xr


def sort_ds(dataset: xr.Dataset) -> xr.Dataset:
    """Sorts the data values in the dataset

    Ensures that profiles are in the following order:

    * Earlier before later (time will increase)
    * Southerly before northerly (latitude will increase)
    * Westerly before easterly (longitude will increase)

    The two xy sorts are esentially tie breakers for when we are missing "time"

    Inside profiles:

    * Shallower before Deeper (pressure will increase)

    """
    # first make sure everything is sorted by pressure
    # this is being done "manually" here becuase xarray only supports 1D sorting
    pressure = dataset.pressure
    sorted_indicies = np.argsort(pressure.values, axis=1)

    for var in dataset.variables:
        # this check ensures that the variable being sorted
        # shares the first two dims as pressure, but allows for more dims past that
        if dataset[var].dims[slice(0, len(pressure.dims))] == pressure.dims:
            dataset[var][:] = np.take_along_axis(
                dataset[var].values, sorted_indicies, axis=1
            )

    # now we can just use the xarray sorting, which only supports 1D
    return dataset.sortby(["time", "latitude", "longitude"])
