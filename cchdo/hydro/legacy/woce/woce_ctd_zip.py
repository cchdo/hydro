from traceback import print_exc
from logging import getLogger


log = getLogger(__name__)


from libcchdo.util import StringIO
from libcchdo.model.datafile import DataFile
from libcchdo.formats import zip as Zip
from libcchdo.formats.ctd import woce
from libcchdo.formats.formats import (
    get_filename_fnameexts, is_filename_recognized_fnameexts,
    is_file_recognized_fnameexts)


_fname_extensions = ['ct.zip']


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
    """How to read CTD WOCE files from a Zip."""
    zip = Zip.ZeroCommentZipFile(handle, 'r')
    try:
        for file in zip.namelist():
            if 'README' in file or 'DOC' in file: continue
            tempstream = StringIO(zip.read(file))
            ctdfile = DataFile()
            try:
                woce.read(ctdfile, tempstream)
            except Exception, e:
                log.info('Failed to read file %s in %s' % (file, handle))
                print_exc()
                raise e
            self.append(ctdfile)
            tempstream.close()
    finally:
        zip.close()

#def write(self, handle): TODO
#    """How to write CTD WOCE files to a Zip."""
