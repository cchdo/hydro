from csv import reader as csv_reader

# TODO: switch to files().joinpath().open when python 3.8 is dropped
# 2023-04-16
from importlib.resources import open_text
from io import BytesIO
from itertools import zip_longest
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import xarray as xr

from cchdo.hydro import accessors as acc

#
CTD_ZIP_FILE_EXTENSION = "ct.zip"
CTD_FILE_EXTENSION = "ct.txt"
BOTTLE_FILE_EXTENSION = "hy.txt"

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

# machinery from COARDS


def simplest_str(s) -> str:
    """Give the simplest string representation.

    If a float is almost equivalent to an integer, swap out for the integer.
    """
    # if type(s) is float:
    if isinstance(s, float):
        # if fns.equal_with_epsilon(s, int(s)):
        # replace with equivalent numpy call
        if np.isclose(s, int(s), atol=1e-6):
            s = int(s)
    return str(s)


def _pad_station_cast(x: str) -> str:
    """Pad a station or cast identifier out to 5 characters. This is usually
    for use in a file name.

    :param x: a string to be padded
    :type x: str
    """
    return simplest_str(x).rjust(5, "0")


def get_filename(expocode, station, cast, file_ext):
    station = _pad_station_cast(station)
    cast = _pad_station_cast(cast)
    return "{}.{}".format(
        "_".join((expocode, station, cast)),
        file_ext,
    )


# END machinery


def convert_fortran_format_to_c(ffmt: str):
    """Simplistic conversion from Fortran format string to C format string.

    This only operates on F formats.
    """
    if not ffmt:
        return ffmt
    if ffmt.startswith("F"):
        return f"%{ffmt[1:]}f"
    elif ffmt.startswith("I"):
        return f"%{ffmt[1:]}d"
    elif ffmt.startswith("A"):
        return f"%{ffmt[1:]}s"
    elif "," in ffmt:
        # WOCE specifies things like 1X,A7, so only convert the last bit.
        ffmt = ffmt.split(",")[1]
        return convert_fortran_format_to_c(ffmt)
    return ffmt


def get_exwoce_params():
    """Return a dictionary of WOCE parameters allowed for Exchange conversion.

    :return: {'PMNEMON': {'unit_mnemonic': 'WOCE', 'range': [0.0, 10.0], 'format': '%8.3f'}}
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


def writeable_columns(ds: xr.Dataset, is_ctd=False):
    """Return the columns that belong in a WOCE data file."""
    CTD_IGNORE = ["STNNBR", "CASTNO", "SAMPNO"]

    # Filter with whitelist and rewrite format strings to WOCE standard.
    whitelisted_columns = []
    for param, col in ds.cchdo.to_whp_columns(compact=True).items():
        key = param.whp_name
        if key in _UNWRITTEN_COLUMNS:
            continue
        if is_ctd and key in CTD_IGNORE:
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


def columns_and_base_format(dfile, is_ctd=False):
    """Return columns and base format for WOCE fixed column data."""
    columns = writeable_columns(dfile, is_ctd=is_ctd)
    num_qualt = len(list(filter(lambda col: acc.FLAG_NAME in col.attrs, columns)))
    col_format = "{{{0}:>8}}"
    base_format = "".join([col_format.format(iii) for iii in range(len(columns))])
    qualt_colsize = max((len("QUALT#"), num_qualt))
    qualt_format = f"{{}}:>{qualt_colsize}"
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

    for row_d, row_f in zip_longest(zip(*data), zip(*flags), fillvalue=""):
        data_lines.append(base_format.format(*row_d, "".join(row_f)))

    return "".join([record2, record3, record4, *data_lines])


def write_bottle(ds: xr.Dataset):
    """How to write a Bottle WOCE file."""
    # Look through datetime for begin and end dates
    begin_date = np.min(ds.time).dt.strftime("%Y%m%d").values
    end_date = np.max(ds.time).dt.strftime("%Y%m%d").values

    expocodes = "/".join(np.unique(ds.expocode))
    sect_id = "NONE"
    if "section_id" in ds:
        sect_id = "/".join(np.unique(ds.section_id))

    columns, base_format = columns_and_base_format(ds)

    vals = [""] * (len(columns) + 1)
    empty_line = base_format.format(*vals)
    record_len = len(empty_line) - 2

    record_1 = f"EXPOCODE {expocodes:s} WHP-ID {sect_id:s} CRUISE DATES {begin_date} TO {end_date}"

    record_1 += " " * (record_len - len(record_1))
    record_1 += "*"
    record_1 += "\n"

    data = write_data(ds, columns, base_format)
    return "".join([record_1, data]).encode("ascii", "replace")


def write_ctd(ds: xr.Dataset):
    """How to write a CTD WOCE file."""
    # We can only write the CTD file if there is a unique
    # EXPOCODE, STNNBR, and CASTNO in the file.
    if ds.sizes["N_PROF"] != 1:
        raise NotImplementedError("can only write single profile")

    expocode = "/".join(np.unique(ds.expocode))
    section = "NONE"
    if "section_id" in ds:
        section = "/".join(np.unique(ds.section_id))
    station = ds.station[0]
    cast = ds.cast[0]

    columns, base_format = columns_and_base_format(ds, is_ctd=True)

    date = ds.time.dt.strftime("%m%d%y")[0].item()

    record1 = f"EXPOCODE {expocode} WHP-ID {section} DATE {date}"
    # 2 at end of line denotes record 2
    record2 = f"STNNBR {station: >8s} CASTNO {cast: >3d} NO. RECORDS={ds.sizes['N_LEVELS']: >5d}"
    # 3 denotes record 3
    instrument_no = ds.get("instrument_id", ["-9"])[0]
    sampling_rate = ds.get("ctd_sampling_rate", [-9])[0]
    record3 = (
        f"INSTRUMENT NO. {instrument_no: >5s} SAMPLING RATE {sampling_rate:>6.2f} HZ"
    )

    headers = "\n".join([record1, record2, record3])
    data = write_data(ds, columns, base_format)
    return "\n".join([headers, data]).encode("ascii", "replace")


def to_woce(ds: xr.Dataset) -> bytes:
    output_files = {}
    profile_type_nd = np.unique(ds.profile_type)

    if len(profile_type_nd) != 1:
        raise NotImplementedError("Cannot convert mixed profile types to woce")
    profile_type = profile_type_nd.item()

    if profile_type == "B":
        return write_bottle(ds)

    elif profile_type == "C":
        for _, profile in ds.groupby("N_PROF", squeeze=False):
            compact = profile.cchdo.compact_profile()
            data = write_ctd(compact)

            filename = get_filename(
                profile.expocode.item(),
                profile.station.item(),
                profile.cast.item(),
                file_ext=CTD_FILE_EXTENSION,
            )
            output_files[filename] = data

        output_zip = BytesIO()
        with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zipfile:
            for fname, data in output_files.items():
                zipfile.writestr(fname, data)

        output_zip.seek(0)
        return output_zip.read()
    else:
        raise NotImplementedError("Unknown profile_type")
