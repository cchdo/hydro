import xarray as xr
from js import Uint8Array
from pyscript import display

from cchdo.hydro import accessors  # noqa: F401


def logger(msg):
    display(msg, target="log", append=True)


def load_netcdf(array_buffer):
    with open("out.nc", "wb") as f:
        f.write(bytearray(Uint8Array.new(array_buffer)))
    return xr.load_dataset("out.nc")


def make_derived(array_buffer, type):
    logger(type)
    xr = load_netcdf(array_buffer)
    if type == "to_sum":
        return xr.cchdo.to_sum(), "summary.txt"
    if type == "to_woce":
        return xr.cchdo.to_woce(), xr.cchdo.gen_fname("woce")
    if type == "to_coards":
        return xr.cchdo.to_coards(), xr.cchdo.gen_fname("coards")


__export__ = [
    "make_derived",
]
