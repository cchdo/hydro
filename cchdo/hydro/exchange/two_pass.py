import logging
import io
import dataclasses
from collections.abc import Mapping
from typing import Tuple, Dict
from itertools import repeat
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


@dataclasses.dataclass(frozen=True)
class ExchangeInfo:
    stamp: int
    comments_start: int
    comments_end: int
    ctd_header_start: int
    ctd_header_end: int
    params: int
    units: int
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
                ctd_header_start = idx
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
        stamp=stamp,
        comments_start=comments_start,
        comments_end=comments_end,
        ctd_header_start=ctd_header_start,
        ctd_header_end=ctd_header_end,
        params=params,
        units=units,
        data_start=data_start,
        data_end=data_end,
        post_data_start=post_data_start,
        post_data_end=post_data_end,
        _raw_lines=lines,
    )


def _prepare_data_block(data_block: Tuple[str, ...]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(a.strip() for a in line.split(",")) for line in data_block)


def _extract_numeric_precisions(data: npt.ArrayLike) -> np.ndarray:
    """Get the numeric precision of a printed decimal number"""
    numeric_parts = np.char.partition(data, ".")[..., 2]

    # magic number explain: np.char.partition expands each element into a 3-tuple
    # of (pre, sep, post) of some sep, in our case a "." char.
    # We only want the post bits [idx 2] (the number of chars after a decimal seperator)
    # of the last axis.
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

    # data_lines = tuple(data.splitlines())

    file_parts = [_get_parts(tuple(df.splitlines()), ftype=ftype) for df in data]
    log.debug(file_parts)

    params = [param.strip() for param in data_lines[file_parts.params].split(",")]
    # we can have a bunch of empty strings as units, we want these to be
    # None to match what would be in a WHPName object
    units = [
        x if x != "" else None
        for x in [unit.strip() for unit in data_lines[file_parts.units].split(",")]
    ]

    # TODO remove when min pyver is 3.10
    if len(params) != len(set(params)):
        raise ExchangeDuplicateParameterError

    # In initial testing, it was discovered that approx half the ctd files
    # had trailing commas in just the params and units lines
    if params[-1] == "" and units[-1] is None:
        log.warning(
            "Removed trailing empty param/unit pair, this indicates these lines have trailing commas."
        )
        params.pop()
        units.pop()

    # the number of expected columns is just going to be the number of
    # parameter names we see
    column_count = len(params)

    if len(units) != column_count:
        raise ExchangeParameterUnitAlignmentError

    whp_params = _bottle_get_params(zip(params, units))
    whp_flags = _bottle_get_flags(zip(params, units), whp_params)
    whp_errors = _bottle_get_errors(zip(params, units), whp_params)

    # ensure we will read all the columns of file
    if {*whp_params.values(), *whp_flags.values(), *whp_errors.values()} != set(
        range(column_count)
    ):
        raise RuntimeError(
            (
                "Not all of the data columns will be read. "
                "This shouldn't happen and is likely a bug, please include the file that caused "
                "this error to occur."
            )
        )

    data_block_raw = np.array(
        _prepare_data_block(data_lines[file_parts.data]), dtype="U"
    )
    data_block_fill = np.char.startswith(data_block_raw, "-999")
    log.debug("Block size: %s bytes", data_block_raw.nbytes)

    # this is basically the only actual "string op" we need to do
    log.debug("Extracting column print precisions")
    # an earlier implimentation just did got the precisions of the entire block
    # that had some significant impacts on ram usage
    whp_param_precisions = {
        param: _extract_numeric_precisions(data_block_raw[:, idx])
        for param, idx in whp_params.items()
        if param.dtype == "decimal"
    }
    whp_error_precisions = {
        param: _extract_numeric_precisions(data_block_raw[:, idx])
        for param, idx in whp_errors.items()
        if param.dtype == "decimal"
    }

    data_block = np.ma.masked_array(data_block_raw, mask=data_block_fill)

    log.debug("Casting to dtypes")
    # TODO confirm dtype mapping
    # For now it looks like ints need to be floats for nan ability
    # this is how xarray does it with an internal encoding
    dtype_map = {"string": "U", "integer": "float32", "decimal": "float64"}
    dtype_fill_values = {"string": "", "integer": np.nan, "decimal": np.nan}

    whp_param_data = {}
    whp_flag_data = {}
    whp_error_data = {}

    for param, idx in whp_params.items():
        param_data = data_block[:, idx]
        param_data.fill_value = dtype_fill_values[param.dtype]
        whp_param_data[param] = np.ma.filled(param_data.astype(dtype_map[param.dtype]))
    for param, idx in whp_flags.items():
        whp_flag_data[param] = data_block[:, idx].astype("float16")
    for param, idx in whp_errors.items():
        whp_error_data[param] = data_block[:, idx].astype(dtype_map[param.dtype])

    log.debug(whp_param_data.values())

    log.debug("Extracting Exchange File Keys")
    exchange_keys = _get_ex_keys(whp_param_data)

    log.debug("Extracting XYZT Coordinates")
    exchange_xyzt = _get_ex_xyzt(whp_param_data)
