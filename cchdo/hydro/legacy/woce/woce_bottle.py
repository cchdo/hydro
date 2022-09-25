from re import compile as re_compile
from logging import getLogger


log = getLogger(__name__)


from libcchdo.fns import strip_all, uniquify
from libcchdo.model.datafile import Column
from libcchdo.formats import woce
from libcchdo.formats.formats import (
    get_filename_fnameexts, is_filename_recognized_fnameexts,
    is_file_recognized_fnameexts)


_fname_extensions = ['hy.txt', '.sea', '.hyd']


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
    '''How to read a Bottle WOCE file.'''
    # Read Woce Bottle header
    try:
        stamp_line = handle.readline()
        parameters_line = handle.readline()
        units_line = handle.readline()
        asterisk_line = handle.readline()
        self.globals['header'] += '\n'.join(
            [stamp_line, parameters_line, units_line, asterisk_line])
    except Exception, e:
        raise ValueError('Malformed WOCE header in WOCE Bottle file: %s' % e)
    # Get stamp
    stamp = re_compile('EXPOCODE\s*([\w/]+)\s*WHP.?ID\s*([\w/-]+(,[\w/-]+)*)\s*CRUISE DATES\s*(\d{6}) TO (\d{6})\s*(\d{8}\w+)?')
    m = stamp.match(stamp_line)
    if m:
        self.globals['EXPOCODE'] = m.group(1)
        self.globals['SECT_ID'] = strip_all(m.group(2).split(','))
        self.globals['_BEGIN_DATE'] = m.group(4)
        self.globals['_END_DATE'] = m.group(5)
        if len(m.groups()) > 6:
            self.globals['stamp'] = m.groups()[-1] # XXX
        else:
            self.globals['stamp'] = None
    else:
        raise ValueError(
            u'Expected ExpoCode, WHP-ID, CRUISE DATES, and possibly a stamp. '
            'Invalid WOCE record 1.')
    # Validate the parameter line
    if 'STNNBR' not in parameters_line or 'CASTNO' not in parameters_line:
        raise ValueError('Expected STNNBR and CASTNO in parameters record')
    woce.read_data(self, handle, parameters_line, units_line, asterisk_line)
    try:
        self.columns['DATE']
    except KeyError:
        self.columns['DATE'] = Column('DATE')
        self.columns['DATE'].values = [None] * len(self) # XXX
    try:
        self.columns['TIME']
    except KeyError:
        self.columns['TIME'] = Column('TIME')
        self.columns['TIME'].values = [None] * len(self)

    woce.fuse_datetime(self)
    
    self.check_and_replace_parameters()


def write(self, handle):
    """How to write a Bottle WOCE file."""

    # Look through datetime for begin and end dates
    datetimes = self.columns["_DATETIME"].values[:]
    begin_date = 0
    end_date = 0
    if any(datetimes):
        usable_datetimes = filter(None, datetimes)
        begin_date = min(usable_datetimes)
        end_date = max(usable_datetimes)
        begin_date = woce.strftime_woce_date(begin_date)
        end_date = woce.strftime_woce_date(end_date)

    # ensure the cruise identifier columns are globals
    if self['EXPOCODE'].is_global():
        self.globals['EXPOCODE'] = self['EXPOCODE'].values[0]
    if self['SECT_ID'].is_global():
        self.globals['SECT_ID'] = self['SECT_ID'].values[0]
    else:
        sect_ids_uniq = uniquify(self['SECT_ID'].values)
        log.warn(u'Multiple section ids found: {0}'.format(sect_ids_uniq))
        self.globals['SECT_ID'] = '/'.join(sect_ids_uniq)

    columns, base_format = woce.columns_and_base_format(self)

    vals = [''] * (len(columns) + 1)
    empty_line = base_format.format(*vals)
    record_len = len(empty_line) - 2

    record_1 = "EXPOCODE %-s WHP-ID %-s CRUISE DATES %6s TO %6s %-s" % (
        self.globals["EXPOCODE"], self.globals["SECT_ID"], begin_date, end_date,
        self.globals['stamp'])

    record_1 += ' ' * (record_len - len(record_1))
    record_1 += '*'
    record_1 += '\n'

    handle.write(record_1)
    woce.write_data(self, handle, columns, base_format)
