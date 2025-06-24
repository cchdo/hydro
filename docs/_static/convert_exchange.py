import io
import json
import logging
import traceback
from html import escape

from js import Uint8Array, console  # noqa: F401
from pyscript import display

from cchdo.hydro import __version__ as hydro_version
from cchdo.hydro import accessors, read_exchange  # noqa: F401
from cchdo.params import __version__ as params_version


def logger(msg):
    display(msg, target="log", append=True)


class DisplaylHandler(logging.Handler):
    def emit(self, record) -> None:
        logger(self.formatter.format(record))


root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = DisplaylHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


def versions():
    return {
        "hydro_version": hydro_version,
        "params_version": params_version,
    }


def to_netcdf(ex):
    ex.to_netcdf("out.nc")
    with open("out.nc", "rb") as f:
        return f.read()


class Pre:
    def __init__(self, text):
        self.value = text

    def _repr_html_(self):
        return f"<pre>{escape(self.value)}</pre>"


def to_xarray(array_buffer, checks):
    checks = json.loads(checks)
    logger(checks)
    logger("to_xarray called")
    bytes = bytearray(Uint8Array.new(array_buffer))
    logger("got bytes")
    ex_bytes = io.BytesIO(bytes)
    try:
        ex = read_exchange(ex_bytes, checks=checks)
        logger("success! making a netCDF file")
    except ValueError as er:
        display(Pre("".join(traceback.format_exception(er))), target="log", append=True)
        display(er.error_data, target="log", append=True)
        raise  # this is so the promise rejects and the main thread knows what's up
    return to_netcdf(ex), ex.cchdo.gen_fname()


__export__ = ["to_xarray", "versions"]
