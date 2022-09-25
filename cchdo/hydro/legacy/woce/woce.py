from datetime import datetime, date
from collections import OrderedDict
import re
import struct
import os.path
from csv import reader as csv_reader
from logging import getLogger


log = getLogger(__name__)


from libcchdo.util import get_library_abspath
from libcchdo.model.datafile import Column
from libcchdo.fns import (
    Decimal, InvalidOperation, _decimal, in_band_or_none, IncreasedPrecision,
    strip_all, uniquify)


# Where no data is known
FILL_VALUE = -9


ASTERISK_FLAG = '*' * 7


CHARACTER_PARAMETERS = ['STNNBR', 'SAMPNO', 'BTLNBR']


COLUMN_WIDTH = 8
SAFE_COLUMN_WIDTH = COLUMN_WIDTH - 1


UNKNONW_TIME_FILL = '0000'


BOTTLE_FLAGS = {
    1: 'Bottle information unavailable.',
    2: 'No problems noted.',
    3: 'Leaking.',
    4: 'Did not trip correctly.',
    5: 'Not reported.',
    6: ('Significant discrepancy in measured values between Gerard and Niskin '
        'bottles.'),
    7: 'Unknown problem.',
    8: ('Pair did not trip correctly. Note that the Niskin bottle can trip at '
        'an unplanned depth while the Gerard trips correctly and vice versa.'),
    9: 'Samples not drawn from this bottle.',
}


CTD_FLAGS = {
    1: 'Not calibrated',
    2: 'Acceptable measurement',
    3: 'Questionable measurement',
    4: 'Bad measurement',
    5: 'Not reported',
    6: 'Interpolated over >2 dbar interval',
    7: 'Despiked',
    8: 'Not assigned for CTD data',
    9: 'Not sampled',
}


WATER_SAMPLE_FLAGS = {
    1: ('Sample for this measurement was drawn from water bottle but analysis '
        'not received.'),
    2: 'Acceptable measurement.',
    3: 'Questionable measurement.',
    4: 'Bad measurement.',
    5: 'Not reported.',
    6: 'Mean of replicate measurements.',
    7: 'Manual chromatographic peak measurement.',
    8: 'Irregular digital chromatographic peak integration.',
    9: 'Sample not drawn for this measurement from this bottle.',
}


def flag_description(flag_map):
    return ':'.join([':'] + ['%d = %s' % (i + 1, flag_map[i + 1]) for i in \
        range(len(flag_map))] + ["\n"])


BOTTLE_FLAG_DESCRIPTION = flag_description(BOTTLE_FLAGS)


CTD_FLAG_DESCRIPTION = flag_description(CTD_FLAGS)


WATER_SAMPLE_FLAG_DESCRIPTION = ':'.join([':'] + \
    ['%d = %s' % (i + 1, WATER_SAMPLE_FLAGS[i + 1]) for i in \
        range(len(WATER_SAMPLE_FLAGS))] + \
    ["\n"])


def hemisphere_to_coeff(hemisphere):
    """Convert a hemisphere to a multiplier."""
    if hemisphere == 'N' or hemisphere == 'E':
        return 1
    return -1


def woce_dec_lat_to_dec_lat(lattoks):
    """Convert a latitude in decimal + hemisphere to decimal."""
    precision = 3 + len(lattoks)
    with IncreasedPrecision(precision):
        try:
            lat = Decimal(lattoks[0])
        except InvalidOperation:
            return None
        lat *= hemisphere_to_coeff(lattoks[1])
        return lat.quantize(Decimal(10) ** -precision)


def woce_dec_lng_to_dec_lng(lngtoks):
    """Convert a longitude in decimal + hemisphere to decimal."""
    precision = 3 + len(lngtoks)
    with IncreasedPrecision(precision):
        try:
            lng = Decimal(lngtoks[0])
        except InvalidOperation:
            return None
        lng *= hemisphere_to_coeff(lngtoks[1])
        return lng.quantize(Decimal(10) ** -precision)


def woce_lat_to_dec_lat(lattoks):
    '''Convert a latitude in WOCE format to decimal.'''
    precision = 3 + len(lattoks)
    with IncreasedPrecision(precision):
        try:
            lat = int(lattoks[0]) + Decimal(lattoks[1]) / Decimal('60.0')
        except InvalidOperation:
            return None
        lat *= hemisphere_to_coeff(lattoks[2])
        return lat.quantize(Decimal(10) ** -precision)


def woce_lng_to_dec_lng(lngtoks):
    '''Convert a longitude in WOCE format to decimal.'''
    precision = 4 + len(lngtoks)
    with IncreasedPrecision(precision):
        try:
            lng = int(lngtoks[0]) + Decimal(lngtoks[1]) / Decimal('60.0')
        except InvalidOperation:
            return None
        lng *= hemisphere_to_coeff(lngtoks[2])
        return lng.quantize(Decimal(10) ** -precision)


def dec_lat_to_woce_lat(lat):
    '''Convert a decimal latitude to WOCE format.'''
    lat_deg = int(lat)
    lat_dec = abs(lat-lat_deg) * 60
    lat_deg = abs(lat_deg)
    lat_hem = 'S'
    if lat > 0:
        lat_hem = 'N'
    return '%2d %05.2f %1s' % (lat_deg, lat_dec, lat_hem)


def dec_lng_to_woce_lng(lng):
    '''Convert a decimal longitude to WOCE format.'''
    lng_deg = int(lng)
    lng_dec = abs(lng-lng_deg) * 60
    lng_deg = abs(lng_deg)
    lng_hem = 'W'
    if lng > 0 :
        lng_hem = 'E'
    return '%3d %05.2f %1s' % (lng_deg, lng_dec, lng_hem)


def strftime_woce_date(dt):
    return dt.strftime('%Y%m%d')


def strftime_woce_time(dt):
    return dt.strftime('%H%M')


def strftime_woce_date_time(dt):
    if dt is None:
        return (None, None)
    if type(dt) is date:
    	return (strftime_woce_date(dt), None)
    return (strftime_woce_date(dt), strftime_woce_time(dt))


def strptime_woce_date(woce_date):
    if woce_date is None:
        return None

    if '-' in str(woce_date):
        woce_date = str(woce_date).translate(None, '-')
    try:
        i_woce_date = int(woce_date)
        return datetime.strptime('%08d' % i_woce_date, '%Y%m%d').date()
    except (TypeError, ValueError):
        log.warn(u"Malformed date {0!r}. Omitting.".format(woce_date))
        return None


def strptime_woce_time(woce_time):
    try:
        i_woce_time = int(woce_time)
        if i_woce_time >= 2400:
            log.warn(
                u"Illegal time {0:04d} >= 2400. Setting to 0.".format(
                    i_woce_time))
            i_woce_time = 0
        return datetime.strptime('%04d' % i_woce_time, '%H%M').time()
    except (TypeError, ValueError):
        log.warn(u"Illegal time {0}. Setting to 0.".format(woce_time))
        return datetime.strptime('0000', '%H%M').time()


def strptime_woce_date_time(woce_date, woce_time):
    """ Parses WOCE date and time into a datetime or date object.
        Args:
            woce_date - a string representing a WOCE date YYYYMMDD
            woce_time - a string representing a WOCE time HHMM
        Returns:
            There are three non-trivial cases:
            1. DATE and TIME both exist
                datetime object representing the combination of the
                two objects.
            2. DATE exists and TIME does not
                date object representing the date.
            3. DATE does not exist but TIME does
                None
    """
    date = strptime_woce_date(woce_date)
    time = strptime_woce_time(woce_time)
    try:
        return datetime.combine(date, time)
    except TypeError:
        return date


def _bad_column_alignment(parameters):
    """ Determine by looking at any of the values if any column is mis-aligned.
        If there is a space anywhere inside a parameter value it is probably
        the result of a misalignment.
    """
    for i, p in enumerate(parameters):
        space_i = p.find(' ')
        if space_i > -1:
            return [i, (i - 1) * COLUMN_WIDTH + space_i]
    return [-1, -1]


def _remove_char_column(i, *lines):
    return [line[:i] + line[i + 1:] for line in lines]


def _remove_char_columns(cols, *lines):
    for col in cols:
        lines = _remove_char_column(col, *lines)
    return lines


def _warn_broke_character_column_rule(headername, headers):
    for header in headers:
        if len(header) > SAFE_COLUMN_WIDTH:
            log.warn("%s '%s' has too many characters (>%d)." % \
                     (headername, header, SAFE_COLUMN_WIDTH))


def _unpack_line(unpack_str, line, num_param_columns):
    values = strip_all(struct.unpack(unpack_str, line[:num_param_columns * 8]))
    return (values, _bad_column_alignment(values))


re_asterisk = re.compile('\*')


def scan_for_trailing_edges(line, clock_signal=re_asterisk):
    edges = []
    clock = False
    for iii, ccc in enumerate(line):
        if clock_signal.match(ccc):
            clock = True
        elif clock:
            clock = False
            edges.append(iii)
    return edges


def split_on_edges(line, edges):
    values = []
    iii = 0
    for jjj in edges:
        values.append(line[iii:jjj].strip())
        iii = jjj
    return values


def read_data_egee(self, handle, parameters_line, units_line, asterisk_line):
    # num_quality_flags = the number of asterisk-marked columns minus the
    # quality word which is marked in EGEE for no good reason.
    num_quality_flags = len(re.findall('\*+', asterisk_line)) - 1
    num_quality_words = len(parameters_line.split('QUALT'))-1

    log.debug(u'{0} quality flags, {1} quality words'.format(
        num_quality_flags, num_quality_words))

    # The extra 1 in quality_length is for spacing between the columns
    quality_length = num_quality_words * (
        max(len('QUALT#'), num_quality_flags) + 1)
    num_param_columns = len(units_line.split())

    re_notword = re.compile('\S')
    p_edges = scan_for_trailing_edges(parameters_line, re_notword)
    u_edges = scan_for_trailing_edges(units_line, re_notword)
    a_edges = scan_for_trailing_edges(asterisk_line)[:-1]

    edges = [max(uuu, aaa) for uuu, aaa in zip(u_edges, a_edges)]

    parameters = split_on_edges(parameters_line, edges)
    units = split_on_edges(unicode(units_line, encoding='utf8'), edges)
    asterisks = split_on_edges(asterisk_line, edges)

    # rewrite duplicate parameters.
    seen = {}
    for param in parameters:
        try:
            seen[param] += 1
        except KeyError:
            seen[param] = 1
    count = {}
    renamed_parameters = []
    for param in parameters:
        if seen[param] > 1:
            try:
                count[param] += 1
            except KeyError:
                count[param] = ord('A')
            letter = chr(count[param])
            param = '{0}_{1}'.format(param, letter)
        if ' ' in param:
            param = param.replace(' ', '')
        if 'CTDFLUO' == param:
            param = 'FLUOR'
        renamed_parameters.append(param)
    parameters = renamed_parameters

    bad_cols = []

    corrected_units = []
    for unit in units:
        if type(unit) == unicode:
            # mu
            if u'\xb5' in unit:
                unit = unit.replace(u'\xb5', 'U')
        corrected_units.append(unit)
    units = corrected_units

    self.create_columns(parameters, units)

    # Get each data line
    # For data, include the QUALT flags in the edges
    edges.append(len(asterisk_line))

    for iii, line in enumerate(handle):
        line = line.rstrip()
        if not line:
            raise ValueError('Empty lines are not allowed in the data section '
                             'of a WOCE file')

        unpacked = split_on_edges(line, edges)
        _build_columns_for_row(
            self, iii, unpacked, num_quality_words, parameters, asterisks)


def _build_columns_for_row(self, iii, row, num_quality_words, parameters,
                           asterisks):
    # QUALT1 takes precedence
    quality_flags = row[-num_quality_words:]

    # Build up the columns for the line
    flag_i = 0
    for j, parameter in enumerate(parameters):
        datum = row[j].strip()
        datum = in_band_or_none(datum, -9)

        if parameter not in CHARACTER_PARAMETERS:
            try:
                datum = _decimal(datum)
            except Exception, e:
                log.warning(
                    u'Expected numeric data for parameter %r, got %r' % (
                    parameter, datum))

        # Only assign flag if column is flagged.
        if "**" in asterisks[j].strip():
            # TODO should use better detection for asterisks
            try:
                woce_flag = int(quality_flags[0][flag_i])
            except ValueError, e:
                log.error(
                    u'Received bad flag "{}" for {} on record {}'.format(
                    quality_flags[0][flag_i], parameter, iii))
                raise e
            flag_i += 1
            self[parameter].set(iii, datum, woce_flag)
        else:
            self[parameter].set(iii, datum)


def read_data(self, handle, parameters_line, units_line, asterisk_line):
    # num_quality_flags = the number of asterisk-marked columns
    num_quality_flags = len(re.findall('\*{7,8}', asterisk_line))
    num_quality_words = len(parameters_line.split('QUALT'))-1

    log.debug(u'{0} quality flags, {1} quality words'.format(
        num_quality_flags, num_quality_words))

    # The extra 1 in quality_length is for spacing between the columns
    quality_length = num_quality_words * (max(len('QUALT#'),
                                              num_quality_flags) + 1)
    num_param_columns = int(
        (len(parameters_line) - quality_length) / COLUMN_WIDTH)

    # Unpack the column headers
    unpack_str = '8s' * num_param_columns

    bad_cols = []

    parameters, start_bad = _unpack_line(
        unpack_str, parameters_line, num_param_columns)
    while start_bad[0] > -1:
        bad_cols.append(start_bad[1])
        log.error('Bad column alignment starting at character %d' % \
                  start_bad[1])
        # Attempt recovery by removing that column.
        parameters_line, units_line, asterisk_line = \
            _remove_char_column(start_bad[1], parameters_line,
                                units_line, asterisk_line)
        parameters, start_bad = _unpack_line(
            unpack_str, parameters_line, num_param_columns)

    units, _ = _unpack_line(unpack_str, units_line, num_param_columns)
    asterisks, _ = _unpack_line(unpack_str, asterisk_line, num_param_columns)

    # Warn if the header lines break 8 character column rules
    _warn_broke_character_column_rule("Parameter", parameters)
    _warn_broke_character_column_rule("Unit", units)
    _warn_broke_character_column_rule("Asterisks", asterisks)

    # Die if parameters are not unique
    if not parameters == uniquify(parameters):
        raise ValueError(('There were duplicate parameters in the file; '
                          'cannot continue without data corruption.'))

    self.create_columns(parameters, units)

    # Get each data line
    unpack_data_str = unpack_str

    # Add on quality to unpack string
    quality_word_spacing = \
        quality_length / num_quality_words - num_quality_flags
    unpack_str += ('%sx%ss' % (quality_word_spacing, num_quality_flags)) * \
                  num_quality_words

    # Let's be nice and try to handle the case where there's a little extra
    # space before the quality flags but everything else is fairly well-formed.
    # It is possible for there to be lots of space between flags and data, so
    # try a few times.
    original_unpack_str = unpack_str
    savepoint = handle.tell()
    line = handle.readline().rstrip()
    determined_num_columns = False
    tries = 0
    while tries < 5:
        try:
            unpacked = struct.unpack(unpack_str, line)
            determined_num_columns = True
            break
        except struct.error, e:
            expected_len = struct.calcsize(unpack_str)
            log.warn(
                'Data record 0 has length %d (expected %d).' % (
                    len(line), expected_len))
            log.info('There is likely extra columns of space between data '
                     'and flags. Detecting whether this is the case.')
            quality_word_spacing += 1
            tries += 1
            unpack_str = unpack_data_str + \
                ('%sx%ss' % (quality_word_spacing, num_quality_flags)
                ) * num_quality_words
    if not determined_num_columns:
        unpack_str = original_unpack_str
    handle.seek(savepoint)
    log.debug(u'Settled on unpack format: {0!r}'.format(unpack_str))

    for iii, line in enumerate(handle):
        line = _remove_char_columns(bad_cols, line.rstrip())[0]
        if not line:
            raise ValueError('Empty lines are not allowed in the data section '
                             'of a WOCE file')
        try:
            unpacked = struct.unpack(unpack_str, line)
        except struct.error, e:
            expected_len = struct.calcsize(unpack_str)
            log.warn('Data record %d has length %d (expected %d).' % (
                iii, len(line), expected_len))
            raise e

        _build_columns_for_row(
            self, iii, unpacked, num_quality_words, parameters, asterisks)

    # Expand globals into columns TODO?


_UNWRITTEN_COLUMNS = [
    'EXPOCODE', 'SECT_ID', 'LATITUDE', 'LONGITUDE', 'DEPTH', '_DATETIME']


WOCE_PARAMS_FOR_EXWOCE = os.path.join(
    get_library_abspath(), 'resources', 'woce_params_for_exchange_to_woce.csv')


def convert_fortran_format_to_c(ffmt):
    """Simplistic conversion from Fortran format string to C format string.
    This only operates on F formats.

    """
    if not ffmt:
        return ffmt
    if ffmt.startswith('F'):
        return '%{0}f'.format(ffmt[1:])
    elif ffmt.startswith('I'):
        return '%{0}d'.format(ffmt[1:])
    elif ffmt.startswith('A'):
        return '%{0}s'.format(ffmt[1:])
    elif ',' in ffmt:
        # WOCE specifies things like 1X,A7, so only convert the last bit.
        ffmt = ffmt.split(',')[1]
        return convert_fortran_format_to_c(ffmt)
    return ffmt


def get_exwoce_params():
    """Return a dictionary of WOCE parameters allowed for Exchange conversion.

    Returns:
        {'PMNEMON': {
            'unit_mnemonic': 'WOCE', 'range': [0.0, 10.0], 'format': '%8.3f'}}

    """
    reader = csv_reader(open(WOCE_PARAMS_FOR_EXWOCE, 'r'))

    # First line is header
    reader.next()

    params = {}
    for order, row in enumerate(reader):
        if row[-1] == 'x':
            continue
        if not row[1]:
            row[1] = None
        if row[2]:
            prange = map(float, row[2].split(','))
        else:
            prange = None
        if not row[3]:
            row[3] = None
        params[row[0]] = {
            'unit_mnemonic': row[1],
            'range': prange,
            'format': convert_fortran_format_to_c(row[3]),
            'order': order,
        }
    return params


_EXWOCE_PARAMS = get_exwoce_params()


def writeable_columns(dfile):
    """Return the columns that belong in a WOCE data file."""
    columns = dfile.columns.values()

    # Filter with whitelist and rewrite format strings to WOCE standard.
    whitelisted_columns = []
    for col in columns:
        key = col.parameter.mnemonic_woce()
        if key in _UNWRITTEN_COLUMNS:
            continue
        if key not in _EXWOCE_PARAMS:
            continue
        info = _EXWOCE_PARAMS[key]
        fmt = info['format']
        if fmt:
            col.parameter.format = fmt
        col.parameter.display_order = info['order']
        whitelisted_columns.append(col)
    return sorted(
        whitelisted_columns, key=lambda col: col.parameter.display_order)


def columns_and_base_format(dfile):
    """Return columns and base format for WOCE fixed column data.

    """
    columns = writeable_columns(dfile)
    num_qualt = len(filter(lambda col: col.is_flagged_woce(), columns))
    col_format = '{{{0}:>8}}'
    base_format = ''.join([col_format.format(iii) for iii in range(len(columns))])
    qualt_colsize = max((len("QUALT#"), num_qualt))
    qualt_format = "{{0}}:>{0}".format(qualt_colsize)
    base_format += ' {' + qualt_format.format(len(columns)) + "}\n"
    return columns, base_format


def truncate_row(lll):
    """Return a new row where all items are less than or equal to column width.

    Warnings will be given for any truncations.

    """
    truncated = []
    for xxx in lll:
        if len(xxx) > COLUMN_WIDTH:
            trunc = xxx[:COLUMN_WIDTH]
            log.warn(u'Truncated {0!r} to {1!r} because longer than {2} '
                     'characters.'.format(xxx, trunc, COLUMN_WIDTH))
            xxx = trunc
        truncated.append(xxx)
    return truncated


def write_data(self, handle, columns, base_format):
    """Write WOCE data in fixed width columns.

    columns and base_format should be obtained from 
    columns_and_base_format()

    """
    def parameter_name_of (column, ):
        return column.parameter.mnemonic_woce()

    def units_of (column, ):
        if column.parameter.units:
            return column.parameter.units.mnemonic
        else:
            return ''

    def quality_flags_of (column, ):
        return ASTERISK_FLAG if column.is_flagged_woce() else ""

    all_headers = map(parameter_name_of, columns)
    all_units = map(units_of, columns)
    all_asters = map(quality_flags_of, columns)

    all_headers.append("QUALT1")
    all_units.append("*")
    all_asters.append("*")

    handle.write(base_format.format(*truncate_row(all_headers)))
    handle.write(base_format.format(*truncate_row(all_units)))
    handle.write(base_format.format(*truncate_row(all_asters)))

    for i in range(len(self)):
        values = []
        flags = []
        for column in columns:
            format = column.parameter.format
            try:
                if column[i]:
                    formatted_value = format % column[i]
                else:
                    formatted_value = format % FILL_VALUE
            except TypeError:
                formatted_value = column[i]
                log.warn(u'Invalid WOCE format for {0} to {1!r}. '
                    'Treating as string.'.format(
                    column.parameter, formatted_value))

            if len(formatted_value) > COLUMN_WIDTH:
                extra = len(formatted_value) - COLUMN_WIDTH
                leading_extra = formatted_value[:extra]
                if len(leading_extra.strip()) == 0:
                    formatted_value = formatted_value[extra:]
                else:
                    old_value = formatted_value
                    formatted_value = formatted_value[:-extra]
                    log.warn(u'Truncated {0!r} to {1} for {2} '
                             'row {3}'.format(old_value, formatted_value,
                                              column.parameter.name, i))

            values.append(formatted_value)
            if column.is_flagged_woce():
                flags.append(str(column.flags_woce[i]))

        values.append("".join(flags))
        handle.write(base_format.format(*values))


def fuse_datetime_globals(file):
    """Fuse a file's "DATE" and "TIME" globals into a "_DATETIME" global.

    There are three cases:
    1. DATE and TIME both exist
        A datetime object is inserted representing the combination
        of the two objects.
    2. DATE exists and TIME does not
        A date object is inserted only representing the date.
    3. DATE does not exist but TIME does
        None is inserted because date is required.

    Arguments:
    file - a DataFile object

    """
    date = file.globals['DATE']
    time = file.globals['TIME']
    file.globals['_DATETIME'] = strptime_woce_date_time(date, time)
    del file.globals['DATE']
    del file.globals['TIME']


def fuse_datetime_columns(file):
    """ Fuses a file's "DATE" and "TIME" columns into a "_DATETIME" column.
        There are three cases:
        1. DATE and TIME both exist
            A datetime object is inserted representing the combination
            of the two objects.
        2. DATE exists and TIME does not
            A date object is inserted only representing the date.
        3. DATE does not exist but TIME does
            None is inserted because date is required.

        Arg:
            file - a DataFile object
    """
    try:
        dates = file['DATE'].values
    except KeyError:
        log.error(u'No DATE column is present.')
        return

    try:
        times = file['TIME'].values
    except KeyError:
        log.warn(u'No TIME column is present.')

    file['_DATETIME'] = Column('_DATETIME')
    file['_DATETIME'].values = [strptime_woce_date_time(*x) for x in zip(
        dates, times)]
    del file['DATE']
    del file['TIME']


def fuse_datetime(file):
    try:
        fuse_datetime_globals(file)
    except KeyError:
        fuse_datetime_columns(file)


def split_datetime_globals(file):
    """Split a file's "_DATETIME" global into "DATE" and "TIME" globals.

    Refer to split_datetime_columns for logic cases.

    Arguments:
        file - a DataFile object

    """
    sdate, stime = strftime_woce_date_time(file.globals['_DATETIME'])
    if not stime:
        stime = UNKNONW_TIME_FILL
    file.globals['DATE'] = sdate
    file.globals['TIME'] = stime
    del file.globals['_DATETIME']


def split_datetime_columns(file):
    """ Splits a file's "_DATETIME" columns into "DATE" and "TIME" columns.

        There are three cases:
        1. datetime
            DATE and TIME are populated appropriately.
        2. date
            Only DATE is populated.
        3. None
            Both DATE and TIME are None

        If there are absolutely no TIMEs in the file the TIME column is not
        kept.

        Arg:
            file - a DataFile object
    """
    dtimecol = file['_DATETIME']
    date = file['DATE'] = Column('DATE')
    time = file['TIME'] = Column('TIME')
    for dtime in dtimecol.values:
        if dtime:
            date.append(strftime_woce_date(dtime))
            if type(dtime) is datetime:
                time.append(strftime_woce_time(dtime))
        else:
            date.append(None)
            time.append(None)
    del file['_DATETIME']

    if not any(file['TIME'].values):
    	file['TIME'].values = [UNKNONW_TIME_FILL] * len(file['TIME'])


def split_datetime(file):
    try:
        split_datetime_globals(file)
    except KeyError:
        try:
            split_datetime_columns(file)
        except KeyError:
            log.warn(u'Unable to split non-existant _DATETIME column')


_MSG_NO_STN_CAST_PAIR = (u'The station cast pair ({}, {}) was not found in '
                         'the summary file.')


def combine(woce_file, sum_file):
    """Combines the given WOCE file with the Summary WOCE file.

    This entails merging the datetime, location, and depth data based on station
    and cast.

    The resulting DataFile contains most of the information from both files.

    """
    headers = sum_file.get_property_for_columns(lambda c: c.parameter.name)

    if woce_file.globals.get('STNNBR', None) is not None:
        # This is probably a CTD file.
        station = woce_file.globals.get('STNNBR')
        cast = woce_file.globals.get('CASTNO')
        try:
            sum_file_index = sum_file.index(station, cast)
        except ValueError, e:
            log.error(_MSG_NO_STN_CAST_PAIR.format(station, cast))
            raise e
        values = sum_file.get_property_for_columns(lambda c: c[sum_file_index])

        info = dict(zip(headers, values))
        woce_file.globals['EXPOCODE'] = info['EXPOCODE']
        woce_file.globals['SECT_ID'] = info['SECT_ID']
        woce_file.globals['_DATETIME'] = info['_DATETIME']
        woce_file.globals['LATITUDE'] = info['LATITUDE']
        woce_file.globals['LONGITUDE'] = info['LONGITUDE']
        woce_file.globals['DEPTH'] = info['DEPTH']
    else:
        # This is probably a Bottle file.
        station_col = woce_file['STNNBR']
        cast_col = woce_file['CASTNO']

        woce_file.ensure_column('EXPOCODE')
        woce_file.ensure_column('SECT_ID')
        woce_file.ensure_column('_DATETIME')
        woce_file.ensure_column('LATITUDE')
        woce_file.ensure_column('LONGITUDE')
        woce_file.ensure_column('DEPTH')

        for i in range(len(woce_file)):
            station = station_col[i]
            cast = cast_col[i]
            try:
                sum_file_index = sum_file.index(station, cast)
            except ValueError, e:
                log.error(_MSG_NO_STN_CAST_PAIR.format(station, cast))
                raise e
            values = sum_file.get_property_for_columns(
                lambda c: c[sum_file_index])

            info = dict(zip(headers, values))
            woce_file['EXPOCODE'][i] = info['EXPOCODE']
            woce_file['SECT_ID'][i] = info['SECT_ID']
            woce_file['_DATETIME'][i] = info['_DATETIME']
            woce_file['LATITUDE'][i] = info['LATITUDE']
            woce_file['LONGITUDE'][i] = info['LONGITUDE']
            woce_file['DEPTH'][i] = info['DEPTH']
        woce_file.globals['header'] = ''
