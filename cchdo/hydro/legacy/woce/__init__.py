from csv import reader as csv_reader

# TODO: switch to files().joinpath().open when python 3.8 is dropped
# 2023-04-16
from importlib.resources import open_text

import numpy as np
import xarray as xr

from ... import accessors as acc

FILL_VALUE = -9

ASTERISK_FLAG = "*" * 7

CHARACTER_PARAMETERS = ["STNNBR", "SAMPNO", "BTLNBR"]

COLUMN_WIDTH = 8
SAFE_COLUMN_WIDTH = COLUMN_WIDTH - 1

UNKNONW_TIME_FILL = "0000"

BOTTLE_FLAGS = {
    1: "Bottle information unavailable.",
    2: "No problems noted.",
    3: "Leaking.",
    4: "Did not trip correctly.",
    5: "Not reported.",
    6: (
        "Significant discrepancy in measured values between Gerard and Niskin "
        "bottles."
    ),
    7: "Unknown problem.",
    8: (
        "Pair did not trip correctly. Note that the Niskin bottle can trip at "
        "an unplanned depth while the Gerard trips correctly and vice versa."
    ),
    9: "Samples not drawn from this bottle.",
}

CTD_FLAGS = {
    1: "Not calibrated",
    2: "Acceptable measurement",
    3: "Questionable measurement",
    4: "Bad measurement",
    5: "Not reported",
    6: "Interpolated over >2 dbar interval",
    7: "Despiked",
    8: "Not assigned for CTD data",
    9: "Not sampled",
}


WATER_SAMPLE_FLAGS = {
    1: (
        "Sample for this measurement was drawn from water bottle but analysis "
        "not received."
    ),
    2: "Acceptable measurement.",
    3: "Questionable measurement.",
    4: "Bad measurement.",
    5: "Not reported.",
    6: "Mean of replicate measurements.",
    7: "Manual chromatographic peak measurement.",
    8: "Irregular digital chromatographic peak integration.",
    9: "Sample not drawn for this measurement from this bottle.",
}


def flag_description(flag_map):
    return ":".join(
        [":"]
        + ["%d = %s" % (i + 1, flag_map[i + 1]) for i in range(len(flag_map))]
        + ["\n"]
    )


BOTTLE_FLAG_DESCRIPTION = flag_description(BOTTLE_FLAGS)

CTD_FLAG_DESCRIPTION = flag_description(CTD_FLAGS)

WATER_SAMPLE_FLAG_DESCRIPTION = ":".join(
    [":"]
    + [
        "%d = %s" % (i + 1, WATER_SAMPLE_FLAGS[i + 1])
        for i in range(len(WATER_SAMPLE_FLAGS))
    ]
    + ["\n"]
)

_UNWRITTEN_COLUMNS = [
    "EXPOCODE",
    "SECT_ID",
    "LATITUDE",
    "LONGITUDE",
    "DEPTH",
    "_DATETIME",
]


def convert_fortran_format_to_c(ffmt: str):
    """Simplistic conversion from Fortran format string to C format string.
    This only operates on F formats.

    """
    if not ffmt:
        return ffmt
    if ffmt.startswith("F"):
        return "%{0}f".format(ffmt[1:])
    elif ffmt.startswith("I"):
        return "%{0}d".format(ffmt[1:])
    elif ffmt.startswith("A"):
        return "%{0}s".format(ffmt[1:])
    elif "," in ffmt:
        # WOCE specifies things like 1X,A7, so only convert the last bit.
        ffmt = ffmt.split(",")[1]
        return convert_fortran_format_to_c(ffmt)
    return ffmt


def get_exwoce_params():
    """Return a dictionary of WOCE parameters allowed for Exchange conversion.

    Returns:
        {'PMNEMON': {
            'unit_mnemonic': 'WOCE', 'range': [0.0, 10.0], 'format': '%8.3f'}}

    """
    with open_text(
        "cchdo.hydro.legacy.woce", "woce_params_for_exchange_to_woce.csv"
    ) as params:
        reader = csv_reader(params)
        params = {}
        for order, row in enumerate(reader):
            # First line is header
            if order == 0:
                continue

            if row[-1] == "x":
                continue
            if not row[1]:
                row[1] = None
            if row[2]:
                prange = list(map(float, row[2].split(",")))
            else:
                prange = None
            if not row[3]:
                row[3] = None
            params[row[0]] = {
                "unit_mnemonic": row[1],
                "range": prange,
                "format": convert_fortran_format_to_c(row[3]),
                "order": order,
            }
        return params


_EXWOCE_PARAMS = get_exwoce_params()


def writeable_columns(ds: xr.Dataset):
    """Return the columns that belong in a WOCE data file."""

    # Filter with whitelist and rewrite format strings to WOCE standard.
    whitelisted_columns = []
    for param, col in ds.cchdo.to_whp_columns(compact=True).items():
        key = param.whp_name
        if key in _UNWRITTEN_COLUMNS:
            continue
        if key not in _EXWOCE_PARAMS:
            continue
        info = _EXWOCE_PARAMS[key]
        fmt = info["format"]
        if fmt:
            col.attrs["cchdo.hydro._format"] = fmt
        col.attrs["cchdo.hydro._display_order"] = info["order"]
        whitelisted_columns.append(col)
    return sorted(
        whitelisted_columns, key=lambda col: col.attrs["cchdo.hydro._display_order"]
    )


def columns_and_base_format(dfile):
    """Return columns and base format for WOCE fixed column data."""
    columns = writeable_columns(dfile)
    num_qualt = len(list(filter(lambda col: acc.FLAG_NAME in col.attrs, columns)))
    col_format = "{{{0}:>8}}"
    base_format = "".join([col_format.format(iii) for iii in range(len(columns))])
    qualt_colsize = max((len("QUALT#"), num_qualt))
    qualt_format = "{{0}}:>{0}".format(qualt_colsize)
    base_format += " {" + qualt_format.format(len(columns)) + "}\n"
    return columns, base_format


def truncate_row(lll):
    """Return a new row where all items are less than or equal to column width.

    Warnings will be given for any truncations.

    """
    truncated = []
    for xxx in lll:
        if len(xxx) > COLUMN_WIDTH:
            trunc = xxx[:COLUMN_WIDTH]
            # jlog.warn(u'Truncated {0!r} to {1!r} because longer than {2} '
            # j         'characters.'.format(xxx, trunc, COLUMN_WIDTH))
            xxx = trunc
        truncated.append(xxx)
    return truncated


def write_data(ds, columns, base_format):
    """Write WOCE data in fixed width columns.

    columns and base_format should be obtained from
    columns_and_base_format()

    """

    def parameter_name_of(column):
        return column.attrs["whp_name"]

    def units_of(column):
        if "whp_unit" in column.attrs:
            return column.attrs["whp_unit"]
        else:
            return ""

    def quality_flags_of(column):
        return ASTERISK_FLAG if acc.FLAG_NAME in column.attrs else ""

    all_headers = list(map(parameter_name_of, columns))
    all_units = list(map(units_of, columns))
    all_asters = list(map(quality_flags_of, columns))

    all_headers.append("QUALT1")
    all_units.append("*")
    all_asters.append("*")

    record2 = base_format.format(*truncate_row(all_headers))
    record3 = base_format.format(*truncate_row(all_units))
    record4 = base_format.format(*truncate_row(all_asters))

    data_lines = []
    # for i in range(ds.dims["N_LEVELS"]):
    #    values = []
    #    flags = []
    #    for column in columns:
    #        format_str = column.attrs.get("cchdo.hydro._format", "%8.f")
    #        try:
    #            formatted_value = format_str % column[i].item()
    #            formatted_value = format_str % FILL_VALUE
    #        except TypeError:
    #            formatted_value = column[i]
    #            #log.warn(u'Invalid WOCE format for {0} to {1!r}. '
    #            #    'Treating as string.'.format(
    #            #    column.parameter, formatted_value))

    #        if len(formatted_value) > COLUMN_WIDTH:
    #            extra = len(formatted_value) - COLUMN_WIDTH
    #            leading_extra = formatted_value[:extra]
    #            if len(leading_extra.strip()) == 0:
    #                formatted_value = formatted_value[extra:]
    #            else:
    #                old_value = formatted_value
    #                formatted_value = formatted_value[:-extra]
    #            #    log.warn(u'Truncated {0!r} to {1} for {2} '
    #            #             'row {3}'.format(old_value, formatted_value,
    #            #                              column.parameter.name, i))

    #        values.append(formatted_value)
    #        #if acc.FLAG_NAME in column.attrs:
    #        #    flags.append(str(column.attrs[acc.FLAG_NAME][i]))

    #    values.append("".join(flags))
    #    data_lines.append(base_format.format(*values))

    data = []
    flags = []
    for column in columns:
        format_str = column.attrs.get("cchdo.hydro._format", "%s")
        str_column = list(
            map(lambda x: format_str % x, column.fillna(FILL_VALUE).to_numpy())
        )
        for i, d in enumerate(str_column):
            if len(d) > COLUMN_WIDTH:
                extra = len(d) - COLUMN_WIDTH
                leading_extra = d[:extra]
                if len(leading_extra.strip()) == 0:
                    d = d[extra:]
                else:
                    d = d[:-extra]

                str_column[i] = d
        data.append(str_column)

        if acc.FLAG_NAME in column.attrs:
            flags.append(
                list(
                    map(
                        lambda x: "%1i" % x,
                        column.attrs[acc.FLAG_NAME].fillna(9).to_numpy(),
                    )
                )
            )

    for row_d, row_f in zip(zip(*data), zip(*flags)):
        data_lines.append(base_format.format(*row_d, "".join(row_f)))

    return "".join([record2, record3, record4, *data_lines])


def write_bottle(ds: xr.Dataset):
    """How to write a Bottle WOCE file."""

    # Look through datetime for begin and end dates
    begin_date = np.min(ds.time).dt.strftime("%Y%m%d")
    end_date = np.max(ds.time).dt.strftime("%Y%m%d")

    # This is just a NOOP now
    # ensure the cruise identifier columns are globals
    # if self['EXPOCODE'].is_global():
    #    self.globals['EXPOCODE'] = self['EXPOCODE'].values[0]
    # if self['SECT_ID'].is_global():
    #    self.globals['SECT_ID'] = self['SECT_ID'].values[0]
    # else:
    #    sect_ids_uniq = uniquify(self['SECT_ID'].values)
    #    log.warn(u'Multiple section ids found: {0}'.format(sect_ids_uniq))
    #    self.globals['SECT_ID'] = '/'.join(sect_ids_uniq)

    expocodes = "/".join(np.unique(ds.expocode))
    sect_id = "NONE"
    if "section_id" in ds:
        sect_id = "/".join(np.unique(ds.section_id))

    columns, base_format = columns_and_base_format(ds)

    vals = [""] * (len(columns) + 1)
    empty_line = base_format.format(*vals)
    record_len = len(empty_line) - 2

    record_1 = f"EXPOCODE {expocodes:s} WHP-ID {sect_id:s} CRUISE DATES {begin_date} TO {end_date} STAMP"

    record_1 += " " * (record_len - len(record_1))
    record_1 += "*"
    record_1 += "\n"

    data = write_data(ds, columns, base_format)
    return "".join([record_1, data])
