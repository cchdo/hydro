from importlib.metadata import PackageNotFoundError, version

__version__: str = "999"

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass

from appdirs import AppDirs

_hydro_appdirs = AppDirs("edu.ucsd.cchdo.hydro")

from .exchange import read_csv, read_exchange

__all__ = ["read_exchange", "read_csv"]
