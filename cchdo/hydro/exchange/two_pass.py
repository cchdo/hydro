import logging
import io
import dataclasses
from collections.abc import Mapping
from typing import Tuple, Dict
from itertools import repeat
from functools import cached_property
from zipfile import ZipFile, is_zipfile, is_zipfile

from cchdo.params import WHPName, WHPNames

import numpy as np
import numpy.typing as npt
import pandas as pd

from .exceptions import (
    ExchangeEncodingError,
    ExchangeBOMError,
    ExchangeLEError,
    ExchangeMagicNumberError,
    ExchangeDuplicateParameterError,
    ExchangeParameterUnitAlignmentError,
    ExchangeDataPartialCoordinateError,
)
from .containers import FileType, ExchangeCompositeKey

from .io import (
    _bottle_get_params,
    _bottle_get_flags,
    _bottle_get_errors,
    _ctd_get_header,
)

log = logging.getLogger(__name__)

EXPOCODE = WHPNames["EXPOCODE"]
STNNBR = WHPNames["STNNBR"]
CASTNO = WHPNames["CASTNO"]

## tmp for new implimentatio
@dataclasses.dataclass(frozen=True)
class ExchangeXYZT(Mapping):
    x: float  # Longitude
    y: float  # Latitude
    z: float  # Pressure
    t: np.datetime64  # Time obviously...

    CTDPRS = WHPNames[("CTDPRS", "DBAR")]
    DATE = WHPNames[("DATE", None)]
    TIME = WHPNames[("TIME", None)]
    LATITUDE = WHPNames[("LATITUDE", None)]
    LONGITUDE = WHPNames[("LONGITUDE", None)]

    WHP_PARAMS: tuple = (
        CTDPRS,
        DATE,
        TIME,
        LATITUDE,
        LONGITUDE,
    )

    TEMPORAL_PARAMS = (DATE, TIME)

    CF_AXIS = {
        DATE: "T",
        TIME: "T",
        LATITUDE: "Y",
        LONGITUDE: "X",
        CTDPRS: "Z",
    }

    def __repr__(self):

        return (
            f"ExchangeXYZT("
            f"x={self.x} "
            f"y={self.y} "
            f"z={self.z} "
            f"t={self.t!r})"
        )

    def __post_init__(self):
        if not all(
            [
                self.x is not None,
                self.y is not None,
                self.z is not None,
            ]
        ):
            raise ExchangeDataPartialCoordinateError(self)

    @property
    def _mapping(self):
        return (
            {
                self.LONGITUDE: self.x,
                self.LATITUDE: self.y,
                self.CTDPRS: self.z,
                self.TIME: self._time_part,
                self.DATE: self._date_part,
            },
        )

    def __eq__(self, other):
        return (self.x, self.y, self.z, self.t) == (other.x, other.y, other.z, other.t)

    def __lt__(self, other):
        """We will consider the following order:
        * A later coordiante is greater than an earlier one
        * A deeper coordinate is greater than a shallower one
        * A more easternly coordinate is greater than a more westerly one
        * A more northernly coordinate is greater than a more southerly one
        The first two points should get most of the stuff we care about sorted
        """
        return (self.t, self.z, self.x, self.y) < (
            other.t,
            other.z,
            other.x,
            other.y,
        )

    def __getitem__(self, key):
        return self._mapping[key]

    def __iter__(self):
        for key in self._mapping:
            yield key

    def __len__(self):
        return len(self._mapping)

    @property
    def _time_part(self):
        if self.t.dtype.name == "datetime64[D]":
            return None
        return pd.Timestamp(self.t).to_pydatetime().time()

    @property
    def _date_part(self):
        return pd.Timestamp(self.t).to_pydatetime().date()


@dataclasses.dataclass
class ExchangeInfo:
    stamp_line: int
    comments_start: int
    comments_end: int
    ctd_header_start: int
    ctd_header_end: int
    params_line: int
    units_line: int
    data_start: int
    data_end: int
    post_data_start: int
    post_data_end: int
    _raw_lines: Tuple[str, ...] = dataclasses.field(repr=False)

    @property
    def comments(self):
        return slice(self.comments_start, self.comments_end)

    @property
    def data(self):
        return slice(self.data_start, self.data_end)

    @property
    def post_data(self):
        return slice(self.post_data_start, self.post_data_end)

    @property
    def ctd_headers(self):
        slc = slice(self.ctd_header_start, self.ctd_header_end)
        return dict([_ctd_get_header(line) for line in self._raw_lines[slc]])

    @cached_property
    def params(self):
        ctd_params = self.ctd_headers.keys()
        return [
            param.strip()
            for param in [*ctd_params, *self._raw_lines[self.params_line].split(",")]
        ]

    @cached_property
    def units(self):
        # we can have a bunch of empty strings as units, we want these to be
        # None to match what would be in a WHPName object
        ctd_units = [None for _ in self.ctd_headers]
        return [
            x if x != "" else None
            for x in [
                *ctd_units,
                *[unit.strip() for unit in self._raw_lines[self.units_line].split(",")],
            ]
        ]

    @cached_property
    def whp_params(self):
        # TODO remove when min pyver is 3.10
        if len(self.params) != len(set(self.params)):
            raise ExchangeDuplicateParameterError

        # In initial testing, it was discovered that approx half the ctd files
        # had trailing commas in just the params and units lines
        if self.params[-1] == "" and self.units[-1] is None:
            log.warning(
                "Removed trailing empty param/unit pair, this indicates these lines have trailing commas."
            )
            self.params.pop()
            self.units.pop()

        # the number of expected columns is just going to be the number of
        # parameter names we see
        column_count = len(self.params)

        if len(self.units) != column_count:
            raise ExchangeParameterUnitAlignmentError

        return _bottle_get_params(zip(self.params, self.units))

    @cached_property
    def whp_flags(self):
        self.whp_params
        return _bottle_get_flags(zip(self.params, self.units), self.whp_params)

    @cached_property
    def whp_errors(self):
        self.whp_params
        return _bottle_get_errors(zip(self.params, self.units), self.whp_params)

    # TODO cache?
    @property
    def _np_data_block(self):
        _raw_data = tuple(
            tuple((*self.ctd_headers.values(), *line.replace(" ", "").split(",")))
            for line in self._raw_lines[self.data]
        )
        return np.array(_raw_data, dtype="U")

    def finalize(self):
        # TODO clean up this function
        log.debug("Finializing...")
        self.single_profile = any(self.ctd_headers)

        np_db = self._np_data_block

        self.length = np_db.shape[0]
        dtype_map = {"string": "U", "integer": "float32", "decimal": "float64"}

        self.whp_param_cols = {}
        self.whp_flag_cols = {}
        self.whp_error_cols = {}
        self.whp_param_precisions = {}
        self.whp_error_precisions = {}

        for param, idx in self.whp_params.items():
            param_col = np_db[:,idx]
            if param.dtype == "decimal":
                self.whp_param_precisions[param] = _extract_numeric_precisions(param_col)
                fill_spaces = np.char.startswith(param_col, "-999")
                param_col[fill_spaces] = "nan"
            self.whp_param_cols[param] = param_col.astype(dtype_map[param.dtype])
        for param, idx in self.whp_flags.items():
            self.whp_flag_cols[param] = np_db[:,idx].astype("float16")
        for param, idx in self.whp_errors.items():
            param_col = np_db[:,idx]
            if param.dtype == "decimal":
                self.whp_error_precisions[param] = _extract_numeric_precisions(param_col)
            self.whp_error_cols[param] = param_col.astype(dtype_map[param.dtype])

        del self._raw_lines



def _get_parts(lines: Tuple[str, ...], ftype: FileType) -> ExchangeInfo:
    """Figure out the line numbers/indicies of the parts of the exchange file"""
    stamp = 0  # file stamp is always the first line of a valid exchange
    comments_start = 1
    comments_end = 1
    ctd_header_start = 1
    ctd_header_end = 1
    params = 1
    units = 1
    data_start = 1
    data_end = 1
    post_data_start = 1
    post_data_end = 1

    looking_for = "file_stamp"
    ctd_num_headers = 0

    log.debug("Looking for file parts")

    for idx, line in enumerate(lines):
        if looking_for == "file_stamp":
            looking_for = "comments"
            continue

        if looking_for == "comments":
            if line.startswith("#"):
                comments_end = idx + 1
            elif ftype == FileType.CTD:
                looking_for = "ctd_headers"
                param, value = _ctd_get_header(line, dtype=int)
                if param != "NUMBER_HEADERS":
                    raise ValueError()
                ctd_num_headers = value - 1
                ctd_header_start = idx + 1
                continue
            else:
                looking_for = "params"
                continue

        if looking_for == "ctd_headers":
            if ctd_num_headers == 0:
                ctd_header_end = idx
                looking_for = "params"
                continue
            ctd_num_headers -= 1

        if looking_for == "params":
            params = idx - 1
            looking_for = "units"
            continue

        if looking_for == "units":
            units = idx - 1
            data_start = idx
            looking_for = "data"
            continue

        if looking_for == "data":
            if line == "END_DATA":
                data_end = idx

                looking_for = "post_data"
                post_data_start = post_data_end = idx + 1
                continue

        if looking_for == "post_data":
            post_data_end = idx

    return ExchangeInfo(
        stamp_line=stamp,
        comments_start=comments_start,
        comments_end=comments_end,
        ctd_header_start=ctd_header_start,
        ctd_header_end=ctd_header_end,
        params_line=params,
        units_line=units,
        data_start=data_start,
        data_end=data_end,
        post_data_start=post_data_start,
        post_data_end=post_data_end,
        _raw_lines=lines,
    )


def _extract_numeric_precisions(data: npt.ArrayLike) -> np.ndarray:
    """Get the numeric precision of a printed decimal number"""
    # magic number explain: np.char.partition expands each element into a 3-tuple
    # of (pre, sep, post) of some sep, in our case a "." char.
    # We only want the post bits [idx 2] (the number of chars after a decimal seperator)
    # of the last axis.
    numeric_parts = np.char.partition(data, ".")[..., 2]
    str_lens = np.char.str_len(numeric_parts)
    return np.max(str_lens, axis=0)


def _get_ex_keys(data: Dict[WHPName, np.ndarray]):

    # get the col indicies
    # TODO make this a classmethod of ExchangeCompositeKey?
    key_cols = zip(
        data[ExchangeCompositeKey.EXPOCODE],
        data[ExchangeCompositeKey.STNNBR],
        data[ExchangeCompositeKey.CASTNO],
        data[ExchangeCompositeKey.SAMPNO],
    )

    exchange_keys = []
    for idx, (expocode, station, cast, sample) in enumerate(key_cols):
        exchange_keys.append(
            (
                ExchangeCompositeKey(
                    expocode=ExchangeCompositeKey.EXPOCODE.data_type(expocode),
                    station=ExchangeCompositeKey.STNNBR.data_type(station),
                    cast=ExchangeCompositeKey.CASTNO.data_type(cast),
                    sample=ExchangeCompositeKey.SAMPNO.data_type(sample),
                ),
                idx,
            )
        )

    return exchange_keys


def _get_ex_xyzt(data: Dict[WHPName, np.ndarray]):
    # fixup date/time to deal with the ability to _not_ have time just a date
    if ExchangeXYZT.TIME not in data:
        time = pd.to_datetime(data[ExchangeXYZT.DATE], format="%Y%m%d").values.astype(
            "datetime64[D]"
        )
    else:
        cat_time = np.char.add(data[ExchangeXYZT.DATE], data[ExchangeXYZT.TIME])
        time = pd.to_datetime(cat_time, format="%Y%m%d%H%M").values.astype(
            "datetime64[m]"
        )

    cols = zip(
        data[ExchangeXYZT.LONGITUDE],
        data[ExchangeXYZT.LATITUDE],
        data[ExchangeXYZT.CTDPRS],
        time,
    )

    ex_xyzt = []
    for idx, (lon, lat, pres, time) in enumerate(cols):
        ex_xyzt.append((ExchangeXYZT(lon, lat, pres, time), idx))

    return ex_xyzt


def read_exchange(path):

    log.info("Trying to open as local filepath")
    with open(path, "rb") as f:
        data_raw = io.BytesIO(f.read())

    data = []
    if is_zipfile(data_raw):

        data_raw.seek(0)  # is_zipfile moves the "tell" position
        with ZipFile(data_raw) as zf:
            for zipinfo in zf.infolist():
                log.debug("Reading %s", zipinfo)
                try:
                    data.append(zf.read(zipinfo).decode("utf8"))
                except UnicodeDecodeError as error:
                    raise ExchangeEncodingError from error
    else:
        data_raw.seek(0)  # is_zipfile moves the "tell" position
        try:
            data.append(data_raw.read().decode("utf8"))
        except UnicodeDecodeError as error:
            raise ExchangeEncodingError from error

    # cleanup the data_raw to free the memory
    data_raw.close()

    log.info("Checking for BOM")
    if any((df.startswith("\ufeff") for df in data)):
        raise ExchangeBOMError

    log.info("Checking Line Endings")
    if any(("\r" in df for df in data)):
        raise ExchangeLEError

    log.info("Detecting file type")
    if all((df.startswith("BOTTLE") for df in data)):
        ftype = FileType.BOTTLE
    elif all((df.startswith("CTD") for df in data)):
        ftype = FileType.CTD
    else:
        # TODO this is where the "mixed" check is happening now
        raise ExchangeMagicNumberError

    log.info("Found filetype: %s", ftype.name)


    # TODO make the parts object better
    file_parts = [_get_parts(tuple(df.splitlines()), ftype=ftype) for df in data]
    for fp in file_parts:
        fp.finalize()

    if all((fp.single_profile for fp in file_parts)):
        log.debug("CTD mode")
        N_PROF = len(file_parts)
        N_LEVELS = max((fp.length for fp in file_parts))

        log.debug((N_PROF, N_LEVELS))

    elif len(file_parts) == 1:
        log.debug("Bottle Mode...?")

        # I guess assume it's a bottle file?
        bottle_file = file_parts[0]
        expocode = bottle_file.whp_param_cols[EXPOCODE]
        station = bottle_file.whp_param_cols[STNNBR]
        cast = bottle_file.whp_param_cols[CASTNO]

        # need to split up by profiles and _not_ assume the bottles are in order
        # use the actual values to sort things out
        # we don't care what the values are, they just need to work
        log.debug("Grouping Profiles by Key")
        prof_ids = np.char.add(np.char.add(expocode, station), cast.astype("U"))
        unique_profile_ids = np.unique(prof_ids)
        log.debug("Found %s unique profile keys", len(unique_profile_ids))
        profiles = [np.nonzero(prof_ids==prof) for prof in unique_profile_ids]

        for profile in profiles:
            for param, data in bottle_file.whp_param_cols.items():
                log.debug((param, data[profile]))




    # ensure we will read all the columns of file
    #if {*whp_params.values(), *whp_flags.values(), *whp_errors.values()} != set(
    #    range(column_count)
    #):
    #    raise RuntimeError(
    #        (
    #            "Not all of the data columns will be read. "
    #            "This shouldn't happen and is likely a bug, please include the file that caused "
    #            "this error to occur."
    #        )
    #    )