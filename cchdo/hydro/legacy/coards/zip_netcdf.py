"""Common code for manipulating netCDF Zip archives."""
from tempfile import NamedTemporaryFile

from libcchdo.model.datafile import DataFile
from libcchdo.formats import netcdf as nc, zip as Zip
from libcchdo.formats.zip import read as zip_read


def read(self, handle, reader):
    """Generic reader for netCDF files in zip."""
    def is_fname_ok(fname):
        return fname.endswith('.nc')
    zip_read(self, handle, is_fname_ok, reader.read)


def get_identifier_btl(dfile):
    """Return a tuple containing the ExpoCode, station and cast for a BTL file.

    """
    expocode = dfile['EXPOCODE'][0] or 'UNKNOWN'
    station = dfile['STNNBR'][0]
    cast = dfile['CASTNO'][0]
    return (expocode, station, cast)


def get_identifier_ctd(dfile):
    """Return a tuple containing the ExpoCode, station and cast for a CTD file.

    """
    expocode = dfile.globals.get('EXPOCODE', 'UNKNOWN')
    station = dfile.globals.get('STNNBR')
    cast = dfile.globals.get('CASTNO')
    return (expocode, station, cast)


class CountingFileName(object):
    """Returns the next available station or cast id."""
    def __init__(self, extension, identifier_func):
        self.station_i = 0
        self.cast_i = 0
        self.extension = extension
        self.identifier_func = identifier_func

    def get_filename(self, dfile):
        expocode, station, cast = self.identifier_func(dfile)
        if station is None:
            station = self.station_i
            self.station_i += 1
        if cast is None:
            cast = self.cast_i
            self.cast_i += 1
        return nc.get_filename(expocode, station, cast, self.extension)
        

def write(self, handle, extension, writer, get_identifier_func, **kwargs):
    """How to write netCDF files to a Zip.

    get_identifier_func(DataFile) - 
        called to get a tuple containing (expocode, station, cast)

    If no station or cast is available, the writer will simply count up starting
    from 1.

    """
    counting_fname = CountingFileName(extension, get_identifier_func)
    Zip.write(self, handle, writer, counting_fname.get_filename, **kwargs)
