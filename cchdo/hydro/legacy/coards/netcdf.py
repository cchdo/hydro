"""Common utilities that NetCDF handlers need."""


import tempfile
from contextlib import contextmanager, closing
import datetime
from logging import getLogger


log = getLogger(__name__)


try:
    from netCDF4 import Dataset
except ImportError, e:
    raise ImportError('%s\n%s' % (e,
        ("Please install netCDF4. (pip install netCDF4)")))

from libcchdo import fns
from libcchdo.fns import Decimal
from libcchdo.formats import woce
from libcchdo.formats.exchange import parse_type_and_stamp_line
from libcchdo.formats.stamped import read_stamp


QC_SUFFIX = '_QC'
FILE_EXTENSION = 'nc'
EPOCH = datetime.datetime(1980, 1, 1, 0, 0, 0)


STATIC_PARAMETERS_PER_CAST = ('EXPOCODE', 'SECT_ID', 'STNNBR', 'CASTNO',
    '_DATETIME', 'LATITUDE', 'LONGITUDE', 'DEPTH', 'BTLNBR', 'SAMPNO', )


NON_FLOAT_PARAMETERS = ('NUMBER', )


UNKNOWN = 'UNKNOWN'


UNSPECIFIED_UNITS = 'unspecified'


STRLEN = 40


def read_type_and_stamp(fileobj):
    """Only get the file type and stamp line.

    For zipfiles, return the most common stamp and warn if there is more than
    one.

    """
    def reader(fobj):
        with closing(tempfile.NamedTemporaryFile()) as fff:
            fff.write(fobj.read())
            fff.flush()
            nc_file = Dataset(fff.name, 'r')
            try:
                first_line = nc_file.ORIGINAL_HEADER.split('\n', 1)[0]
                return parse_type_and_stamp_line(first_line)
            except (AttributeError):
                return ('', '')
    return read_stamp(fileobj, reader)


def ascii(x):
    return x.encode('ascii', 'replace')


def simplest_str(s):
    """Give the simplest string representation.
       If a float is almost equivalent to an integer, swap out for the
       integer.
    """
    if type(s) is float:
        if fns.equal_with_epsilon(s, int(s)):
            s = int(s)
    return str(s)


def _pad_station_cast(x):
    """Pad a station or cast identifier out to 5 characters. This is usually
       for use in a file name.
       Args:
            x - a string to be padded
    """
    return simplest_str(x).rjust(5, '0')


@contextmanager
def nc_dataset_to_stream(stream, *args, **kwargs):
    """Creates a DataSet and writes it out to the stream when closed."""
    # netcdf library wants to write its own files.
    tmp = tempfile.NamedTemporaryFile()
    nc_file = Dataset(tmp.name, 'w', *args, **kwargs)
    try:
        yield nc_file
    finally:
        nc_file.close()
        stream.write(tmp.read())
        tmp.close()


def get_filename(expocode, station, cast, extension):
    if extension not in ['hy1', 'ctd']:
        log.warn(u'File extension is not recognized.')
    station = _pad_station_cast(station)
    cast = _pad_station_cast(cast)
    return '%s.%s' % ('_'.join((expocode, station, cast, extension)),
                      FILE_EXTENSION, )


def minutes_since_epoch(dt, error=-9):
    if not dt:
        return error
    if type(dt) is datetime.date:
    	dt = datetime.datetime(dt.year, dt.month, dt.day)
    delta = dt - EPOCH
    minutes_in_day = 60 * 24
    minutes_in_seconds = 1.0 / 60
    minutes_in_microseconds = minutes_in_seconds / 1.0e6
    return (delta.days * minutes_in_day + \
            delta.seconds * minutes_in_seconds + \
            delta.microseconds * minutes_in_microseconds)


def check_variable_ranges(nc_file):
    for name, variable in nc_file.variables.items():
        try:
            min = Decimal(str(variable.data_min))
            max = Decimal(str(variable.data_max))
        except AttributeError:
            try:
                min = Decimal(str(variable.valid_min))
                max = Decimal(str(variable.valid_max))
            except AttributeError:
                continue
        for y in variable[:]:
            if fns.isnan(y):
                continue
            x = Decimal(str(y))
            if x < min:
                log.warn('%s too small for %s range (%s, %s)' % \
                         (str(x), name, str(min), str(max)))
            if x > max:
                log.warn('%s too large for %s range (%s, %s)' % \
                         (str(x), name, str(min), str(max)))


def define_dimensions(nc_file, length):
    """Create NetCDF file dimensions."""
    makeDim = nc_file.createDimension
    makeDim('time', 1)
    makeDim('pressure', length)
    makeDim('latitude', 1)
    makeDim('longitude', 1)
    makeDim('string_dimension', STRLEN)


def define_attributes(nc_file, expocode, sect_id, data_type, stnnbr, castno,
                      bottom_depth):
    nc_file.EXPOCODE = expocode
    nc_file.Conventions = 'COARDS/WOCE'
    nc_file.WOCE_VERSION = '3.0'
    nc_file.WOCE_ID = sect_id
    nc_file.DATA_TYPE = data_type
    nc_file.STATION_NUMBER = stnnbr
    nc_file.CAST_NUMBER = castno
    nc_file.BOTTOM_DEPTH_METERS = bottom_depth
    nc_file.Creation_Time = fns.strftime_iso(datetime.datetime.utcnow())


def set_original_header(nc_file, dfile, datatype):
    nc_file.ORIGINAL_HEADER = '\n'.join([
        '{0},{1}'.format(datatype, dfile.globals.get('stamp', '')),
        dfile.globals.get('header', '')])
    

def create_common_variables(nc_file, latitude, longitude, woce_datetime,
                            stnnbr, castno):
    """Add variables to the netcdf file object such as date, time etc."""
    # Coordinate variables

    var_time = nc_file.createVariable('time', 'i', ('time',))
    var_time.long_name = 'time'
    # Java OceanAtlas 5.0.2 requires ISO 8601 with space separator.
    var_time.units = 'minutes since %s' % EPOCH.isoformat(' ')
    var_time.data_min = int(minutes_since_epoch(woce_datetime))
    var_time.data_max = var_time.data_min
    var_time.C_format = '%10d'
    var_time[:] = var_time.data_min

    var_latitude = nc_file.createVariable('latitude', 'f', ('latitude',))
    var_latitude.long_name = 'latitude'
    var_latitude.units = 'degrees_N'
    var_latitude.data_min = float(latitude)
    var_latitude.data_max = var_latitude.data_min
    var_latitude.C_format = '%9.4f'
    var_latitude[:] = var_latitude.data_min

    var_longitude = nc_file.createVariable('longitude', 'f', ('longitude',))
    var_longitude.long_name = 'longitude'
    var_longitude.units = 'degrees_E'
    var_longitude.data_min = float(longitude)
    var_longitude.data_max = var_longitude.data_min
    var_longitude.C_format = '%9.4f'
    var_longitude[:] = var_longitude.data_min

    strs_woce_datetime = woce.strftime_woce_date_time(woce_datetime)

    var_woce_date = nc_file.createVariable('woce_date', 'i', ('time',))
    var_woce_date.long_name = 'WOCE date'
    var_woce_date.units = 'yyyymmdd UTC'
    var_woce_date.data_min = int(strs_woce_datetime[0] or -9)
    var_woce_date.data_max = var_woce_date.data_min
    var_woce_date.C_format = '%8d'
    var_woce_date[:] = var_woce_date.data_min
    
    if strs_woce_datetime[1]:
        var_woce_time = nc_file.createVariable('woce_time', 'i2', ('time',))
        var_woce_time.long_name = 'WOCE time'
        var_woce_time.units = 'hhmm UTC'
        var_woce_time.data_min = int(strs_woce_datetime[1] or -9)
        var_woce_time.data_max = var_woce_time.data_min
        var_woce_time.C_format = '%4d'
        var_woce_time[:] = var_woce_time.data_min
    
    # Hydrographic specific
    
    var_station = nc_file.createVariable('station', 'c', ('string_dimension',))
    var_station.long_name = 'STATION'
    var_station.units = UNSPECIFIED_UNITS
    var_station.C_format = '%s'
    var_station[:] = simplest_str(stnnbr).ljust(len(var_station))
    
    var_cast = nc_file.createVariable('cast', 'c', ('string_dimension',))
    var_cast.long_name = 'CAST'
    var_cast.units = UNSPECIFIED_UNITS
    var_cast.C_format = '%s'
    var_cast[:] = simplest_str(castno).ljust(len(var_cast))


def create_and_fill_data_variables(df, nc_file):
    """Add variables to the netcdf file object that correspond to data."""
    for column in df.sorted_columns():
        parameter = column.parameter
        if not parameter:
            continue

        parameter_name = parameter.mnemonic_woce()
        if parameter_name in STATIC_PARAMETERS_PER_CAST:
            continue

        if parameter_name in NON_FLOAT_PARAMETERS:
            continue

        pname = parameter.name_netcdf
        if not pname:
            log.warn(
                u'No netcdf name for {0}. Using mnemonic {1}.'.format(
                    parameter, parameter.name))
            pname = parameter.name
        if not pname:
            raise AttributeError('No name found for {0}'.format(parameter))
        pname = ascii(pname)

        # XXX HACK
        if pname == 'oxygen1':
            pname = 'oxygen'

        var = nc_file.createVariable(pname, 'f8', ('pressure',))
        var.long_name = pname

        if var.long_name == 'pressure':
            var.positive = 'down'

        units = UNSPECIFIED_UNITS
        if parameter.units:
            units = parameter.units.name
        units = ascii(units)
        var.units = units

        compact_column = filter(None, column)
        if compact_column:
            var.data_min = float(min(compact_column))
            var.data_max = float(max(compact_column))
        else:
            var.data_min = float('-inf')
            var.data_max = float('inf')

        if parameter.format:
            var.C_format = ascii(parameter.format)
        else:
            # TODO TEST this
            log.warn(u"Parameter {0} has no format. defaulting to '%f'".format(
                parameter.name))
            var.C_format = '%f'
        if var.C_format.endswith('s'):
            log.warn(
                u'Parameter {0} does not have a format string acceptable for '
                "numeric data. Defaulting to '%f' to prevent ncdump "
                'segfault.'.format(parameter.name))
            var.C_format = '%f'
        var.WHPO_Variable_Name = parameter_name
        var[:] = column.values

        if column.is_flagged_woce():
            qc_name = pname + QC_SUFFIX
            var.OBS_QC_VARIABLE = qc_name
            vfw = nc_file.createVariable(qc_name, 'i2', ('pressure',))
            vfw.long_name = qc_name + '_flag'
            vfw.units = 'woce_flags'
            vfw.C_format = '%1d'
            vfw[:] = column.flags_woce


@contextmanager
def buffered_netcdf(handle, *args, **kwargs):
    """Buffer netcdf writing to a temporary file before writing it to handle.

    """
    tmp = tempfile.NamedTemporaryFile()
    nc_file = Dataset(tmp.name, *args, **kwargs)

    try:
        yield nc_file
    finally:
        nc_file.close()
        handle.write(tmp.read())
        tmp.close()
