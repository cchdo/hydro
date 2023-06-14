from importlib.metadata import version, PackageNotFoundError

__version__ = "not installed"

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

from appdirs import AppDirs

_hydro_appdirs = AppDirs("edu.ucsd.cchdo.hydro")

from .exchange import read_exchange, read_csv


__all__ = ["read_exchange", "read_csv"]
