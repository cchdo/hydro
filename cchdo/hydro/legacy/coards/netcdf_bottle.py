import datetime
import os
import re
import tempfile
from logging import getLogger


log = getLogger(__name__)


import numpy as np

from libcchdo import fns
from libcchdo.util import memoize
from libcchdo.db.model import std
from libcchdo.formats import netcdf as nc
from libcchdo.formats import woce
from libcchdo.formats.exchange import FILL_VALUE
from libcchdo.formats.formats import (
    get_filename_fnameexts, is_filename_recognized_fnameexts,
    is_file_recognized_fnameexts)


_fname_extensions = ['hy1.nc']


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


@memoize
def nc_bottle_var_to_woce_param():
    return dict(std.session().query(
        std.Parameter.name_netcdf, std.Parameter.name).all())


VARATTRS = frozenset(('time', 'latitude', 'longitude', 'woce_date',
                      'woce_time', 'cast', 'station', ))


def read(self, handle):
    """How to read a Bottle NetCDF file."""
    filename = handle.name
    nc_file = nc.Dataset(filename, 'r')
    
    attrs = nc_file.__dict__
    expocode = attrs.get('EXPOCODE')
    self.globals['header'] = attrs.get('ORIGINAL_HEADER')
    station = attrs.get('STATION_NUMBER').strip()
    cast = attrs.get('CAST_NUMBER').strip()
    bottle_numbers = attrs.get('BOTTLE_NUMBERS', '').split()
    bottle_flags = attrs.get('BOTTLE_QUALITY_CODES', [])[:]
    section_id = attrs.get('WOCE_ID')
    bottom_depth = attrs.get('BOTTOM_DEPTH_METERS')

    vars = nc_file.variables

    time = vars['time'][:][0]
    latitude = vars['latitude'][:][0]
    longitude = vars['longitude'][:][0]
    woce_date = vars['woce_date'][:][0]
    woce_time = vars.get('woce_time', [None])[:][0]
    dtime = woce.strptime_woce_date_time(woce_date, woce_time)

    calculated_time = nc.EPOCH + datetime.timedelta(minutes=int(time))
    # TODO Probably should trust dtime more because it is translated directly
    # from WOCE time.
    if type(dtime) is datetime.date:
    	calculated_time = calculated_time.date()
    if dtime != calculated_time:
        log.warn(('Datetime declarations in Bottle NetCDF file '
                  'do not match (%s, %s)') % (dtime, calculated_time))

    varstation = ''.join(filter(None, vars['station'][:].tolist())).strip()
    varcast = ''.join(filter(None, vars['cast'][:].tolist())).strip()

    if varstation != station:
        log.warn(('Station declarations in Bottle NetCDF file '
                  'do not match (%s, %s)') % (station, varstation))

    if varcast != cast:
        log.warn(('Cast declarations in Bottle NetCDF file '
                  'do not match (%s, %s)') % (cast, varcast))

    # Create global columns if they do not exist
    globals_to_vars = {
        'EXPOCODE': ('', expocode),
        'SECT_ID': ('', section_id),
        'STNNBR': ('', station),
        'CASTNO': ('', cast),
        'DEPTH': ('METERS', bottom_depth),
        '_DATETIME': ('', dtime),
    }
    gs = globals_to_vars.keys()
    self.create_columns(gs)
    self.create_columns(('BTLNBR', ))

    # Fill global columns with data
    dimensions = len(nc_file.dimensions['pressure'])
    vlo = len(self)
    vhi = vlo + dimensions
    for g, var in globals_to_vars.items():
        self[g].values[vlo:vhi] = [var[1]] * dimensions

    self['BTLNBR'].values[vlo:vhi] = bottle_numbers

    # First pass to create columns
    qc_vars = {}
    for name in frozenset(vars.keys()) - VARATTRS:
        variable = vars[name]
        if name.endswith(nc.QC_SUFFIX):
            qc_vars[nc_bottle_var_to_woce_param()[
                name[:-len(nc.QC_SUFFIX)]]] = variable
        else:
            name = nc_bottle_var_to_woce_param().get(name, name)
            
            if name == 'drop':
                continue

            self.create_columns((name, ))
            self[name].values[vlo:vhi] = variable[:].tolist()

            # Quick conversions to uniform data format
            self[name].values[vlo:vhi] = map(
                fns.in_band_or_none,
                self[name].values[vlo:vhi])

    # Second pass to put in flags
    for name, variable in qc_vars.items():
        if name in self.columns:
            self[name].flags_woce[vlo:vhi] = variable[:].tolist()
        else:
            # The column is probably a global
            pass

    # Pad out columns that aren't present in this read to maintain
    # file structure.
    nones = [None for i in range(vlo, vhi)]
    for c in self.columns.values():
        if len(c) < vhi:
            c.values[vlo:vhi] = nones
            if c.is_flagged_woce():
                c.flags_woce[vlo:vhi] = nones
            if c.is_flagged_igoss():
                c.flags_igoss[vlo:vhi] = nones

    nc_file.close()

    self.check_and_replace_parameters()


def _lambda_or_unknown(fn, unknown=nc.UNKNOWN):
    """Attempt to return the result of fn; on error return unknown."""
    try:
        return fn()
    except (KeyError, IndexError):
        return unknown


def _create_common_variables(df, nc_file):
    latitude = df['LATITUDE'][0]
    longitude = df['LONGITUDE'][0]
    # Take the time of the first tripped bottle for the cast, in accordance with
    # WOCE spec.
    woce_datetime = min(df['_DATETIME'])
    stnnbr = df['STNNBR'][0]
    castno = df['CASTNO'][0]
    nc.create_common_variables(
        nc_file, latitude, longitude, woce_datetime, stnnbr, castno)


def _nc_bottom_depth(df):
    try:
        depths = filter(None, df['DEPTH'].values)
        if not depths:
            return FILL_VALUE
        return int(max(depths))
    except (KeyError, AttributeError):
        return FILL_VALUE


def write(self, handle):
    """How to write a Bottle NetCDF file."""
    temp = tempfile.NamedTemporaryFile()
    nc_file = nc.Dataset(temp.name, 'w', format='NETCDF3_CLASSIC')

    nc.define_dimensions(nc_file, len(self))

    # Define dataset attributes
    nc.define_attributes(
        nc_file, 
        _lambda_or_unknown(lambda: self['EXPOCODE'][0]),
        _lambda_or_unknown(lambda: self['SECT_ID'][0]),
        'WOCE Bottle',
        _lambda_or_unknown(lambda: nc.simplest_str(self['STNNBR'][0])),
        _lambda_or_unknown(lambda: nc.simplest_str(self['CASTNO'][0])),
        _nc_bottom_depth(self),
        )

    nc.set_original_header(nc_file, self, 'BOTTLE')

    try:
        bottle_column = self['BTLNBR']
    except KeyError:
        bottle_column = self['SAMPNO']

    nc_file.BOTTLE_NUMBERS = ' '.join(
        map(nc.simplest_str, bottle_column.values))
    if bottle_column.is_flagged_woce():
        # Java OceanAtlas 5.0.2 and possibly before requires bottle quality
        # codes to be shorts.
        btl_quality_codes = \
            np.array(bottle_column.flags_woce).astype(np.int16)
        nc_file.BOTTLE_QUALITY_CODES = btl_quality_codes

    nc_file.WOCE_BOTTLE_FLAG_DESCRIPTION = woce.BOTTLE_FLAG_DESCRIPTION
    nc_file.WOCE_WATER_SAMPLE_FLAG_DESCRIPTION = \
        woce.WATER_SAMPLE_FLAG_DESCRIPTION

    nc.create_and_fill_data_variables(self, nc_file)
    _create_common_variables(self, nc_file)

    nc_file.close()
    handle.write(temp.read())
    temp.close()
