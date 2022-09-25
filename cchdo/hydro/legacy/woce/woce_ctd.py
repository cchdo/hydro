import re
from datetime import datetime

from libcchdo.fns import uniquify
from libcchdo.formats import woce
from libcchdo.formats.formats import (
    get_filename_fnameexts, is_filename_recognized_fnameexts,
    is_file_recognized_fnameexts)


_fname_extensions = ['.ctd']


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
    """How to read a CTD WOCE file."""
    # TODO Out of band values should be converted to None

    # Get record 1
    re_frag_expocode = 'EXPOCODE\s*([\w/]+)\s*'
    re_frag_whpid = 'WHP.?IDS?\s*([\w/-]+(,[\w/-]+)?)\s*'
    re_frag_date = 'DATE\s*(\d{6})(?:\s*1)?'
    re_record1 = re.compile(
        ''.join([re_frag_expocode, re_frag_whpid, re_frag_date]),
        re.IGNORECASE)
    line = handle.readline()
    m = re_record1.match(line)
    if m:
        self.globals['EXPOCODE'] = m.group(1)
        self.globals['SECT_ID'] = m.group(2)
        self.globals['_DATETIME'] = datetime.strptime(
            m.group(len(m.groups())), '%m%d%y')
        self.globals['DATE'], self.globals['TIME'] = \
            woce.strftime_woce_date_time(self.globals['_DATETIME'])
    else:
        # Try record 1 again without WHP-ID
        re_record1 = re.compile(
            ''.join([re_frag_expocode, re_frag_date]), re.IGNORECASE)
        m = re_record1.match(line)
        if m:
            self.globals['EXPOCODE'] = m.group(1)
            self.globals['_DATETIME'] = datetime.strptime(
                m.group(len(m.groups())), '%m%d%y')
            self.globals['DATE'], self.globals['TIME'] = \
                woce.strftime_woce_date_time(self.globals['_DATETIME'])
        else:
            raise ValueError("Invalid record 1 in WOCE CTD file.")

    # Get identifier line
    identifier = re.compile(
        'STNNBR\s*(\d+)\s*CASTNO\s*(\d+)\s*NO\. Records=\s*(\d+)',
        re.IGNORECASE)
    m = identifier.match(handle.readline())
    if m:
        self.globals['STNNBR'] = m.group(1)
        self.globals['CASTNO'] = m.group(2)
    else:
        raise ValueError(("Expected identifiers. Invalid record 2 in "
                          "WOCE CTD file."))

    # Get instrument line
    instrument = re.compile(
        'INSTRUMENT NO.\s*(\d+)\s*SAMPLING RATE\s*(\d+.\d+\s*HZ)',
        re.IGNORECASE)
    m = instrument.match(handle.readline())
    if m:
        self.globals['_INSTRUMENT_NO'] = m.group(1)
        self.globals['_SAMPLING_RATE'] = m.group(2)
    else:
        raise ValueError(("Expected instrument information. "
                          "Invalid record 3 in WOCE CTD file."))
    
    parameters_line = handle.readline()
    units_line = handle.readline()
    asterisk_line = handle.readline()

    woce.read_data(self, handle, parameters_line, units_line, asterisk_line)

    self.check_and_replace_parameters()


def write(self, handle):
    '''How to write a CTD WOCE file.'''
    # We can only write the CTD file if there is a unique
    # EXPOCODE, STNNBR, and CASTNO in the file.
    expocodes = self.globals["EXPOCODE"] #self.expocodes()
    sections = self.globals["SECT_ID"] #uniquify(self.columns['SECT_ID'].values)
    stations = self.globals["STNNBR"] #uniquify(self.columns['STNNBR'].values)
    casts = self.globals["CASTNO"] #uniquify(self.columns['CASTNO'].values)

    #def has_multiple_values(a):
    #    return len(a) is not 1

    #if any(map(has_multiple_values, [expocodes, sections, stations, casts])):
    #  raise ValueError(('Cannot write a multi-ExpoCode, section, station, '
    #                    'or cast WOCE CTD file.'))
    #else:
    #  expocode = expocodes[0]
    #  section = sections[0]
    #  station = stations[0]
    #  cast = casts[0]

    expocode = expocodes # XXX
    section = sections   # XXX
    station = stations   # XXX
    cast = casts         # XXX

    date = int(self.globals['_DATETIME'].strftime('%m%d%y'))

    handle.write('EXPOCODE %-14s WHP-ID %-5s DATE %06d\n' % \
                 (expocode, section, date))
    # 2 at end of line denotes record 2
    handle.write('STNNBR %-8s CASTNO %-3d NO. RECORDS=%-5d%s\n' %
                 (station, int(cast), len(self.columns), ""))
    # 3 denotes record 3
    handle.write('INSTRUMENT NO. %-5s SAMPLING RATE %-6.2f HZ%s\n' %
                 (0, 42.0, ""))
    #handle.write('  CTDPRS  CTDTMP  CTDSAL  CTDOXY  NUMBER QUALT1') # TODO
    #handle.write('    DBAR  ITS-90  PSS-78 UMOL/KG    OBS.      *') # TODO
    #handle.write(' ******* ******* ******* *******              *') # TODO
    #handle.write('     3.0 28.7977 31.8503   209.5      42   2222') # TODO

    woce.write_data(self, handle)
