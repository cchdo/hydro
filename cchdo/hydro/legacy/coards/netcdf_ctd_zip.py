"""CCHDO format for COARDS compliant netCDF files for CTD."""
from libcchdo.formats.ctd import netcdf as ctdnc
from libcchdo.formats import zip_netcdf as zipnc
from libcchdo.formats.formats import (
    get_filename_fnameexts, is_filename_recognized_fnameexts,
    is_file_recognized_fnameexts)


_fname_extensions = ['nc_ctd.zip']


def get_filename(basename):
    """Return the filename for this format given a base filename.

    This is a basic implementation using filename extensions.

    """
    return get_filename_fnameexts(basename, _fname_extensions)


def is_filename_recognized(fname):
    """Return whether the given filename is a match for this file format.

    This is a basic implementation using filename extensions.

    """
    return is_filename_recognized_fnameexts(fname, _fname_extensions)


def is_file_recognized(fileobj):
    """Return whether the file is recognized based on its contents.

    This is a basic non-implementation.

    """
    return is_file_recognized_fnameexts(fileobj, _fname_extensions)


def read(self, handle):
    """How to read CTD NetCDF files from a Zip."""
    zipnc.read(self, handle, ctdnc)


def write(self, handle):
    """How to write CTD NetCDF files to a Zip."""
    zipnc.write(self, handle, 'ctd', ctdnc, zipnc.get_identifier_ctd)
