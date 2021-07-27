from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

from appdirs import AppDirs

_hydro_appdirs = AppDirs("edu.ucsd.cchdo.hydro")
