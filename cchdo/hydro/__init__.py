from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

from appdirs import AppDirs

_hydro_appdirs = AppDirs("edu.ucsd.cchdo.hydro")
