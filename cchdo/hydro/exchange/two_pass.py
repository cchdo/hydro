import logging
import io
import dataclasses
from typing import Tuple

import numpy as np

from .exceptions import (
    ExchangeEncodingError,
    ExchangeBOMError,
    ExchangeLEError,
    ExchangeMagicNumberError,
    ExchangeDuplicateParameterError,
    ExchangeParameterUnitAlignmentError,
)
from .containers import FileType, ExchangeCompositeKey

from .io import _bottle_get_params, _bottle_get_flags, _bottle_get_errors

log = logging.getLogger(__name__)


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
    params = 1
    units = 1
    data_start = 1
    data_end = 1
    post_data_start = 1
    post_data_end = 1

    looking_for = "file_stamp"

    log.debug("Looking for file parts")

    log.debug("Now looking for: %s", looking_for)
    for idx, line in enumerate(lines):
        if looking_for == "file_stamp":
            log.debug("Found %s on line %d", looking_for, idx + 1)
            looking_for = "comments"
            log.debug("Now looking for: %s", looking_for)
            continue

        if looking_for == "comments":
            if line.startswith("#"):
                comments_end = idx + 1
            elif ftype == FileType.CTD:
                log.debug(
                    "Found %s on lines %d to %d",
                    looking_for,
                    comments_start + 1,
                    comments_end + 1,
                )
                looking_for = "ctd_headers"
                log.debug("Now looking for: %s", looking_for)
                continue
            else:
                log.debug(
                    "Found %s on lines %d to %d",
                    looking_for,
                    comments_start + 1,
                    comments_end + 1,
                )
                looking_for = "params"
                log.debug("Now looking for: %s", looking_for)
                continue

        if looking_for == "ctd_headers":
            raise NotImplementedError()

        if looking_for == "params":
            params = idx - 1
            log.debug("Found %s on line %d", looking_for, idx + 1)
            looking_for = "units"
            log.debug("Now looking for: %s", looking_for)
            continue

        if looking_for == "units":
            units = idx - 1
            log.debug("Found %s on line %d", looking_for, idx + 1)
            data_start = idx
            looking_for = "data"
            log.debug("Now looking for: %s", looking_for)
            continue

        if looking_for == "data":
            if line == "END_DATA":
                data_end = idx

                log.debug(
                    "Found %s on lines %d to %d",
                    looking_for,
                    data_start + 1,
                    data_end + 1,
                )

                looking_for = "post_data"
                post_data_start = post_data_end = idx + 1
                log.debug("Now looking for: %s", looking_for)
                continue

        if looking_for == "post_data":
            post_data_end = idx

    if post_data_end == post_data_start:
        log.debug("No post data content")
    else:
        log.debug(
            "Found %s on lines %d to %d",
            looking_for,
            data_start + 1,
            data_end + 1,
        )

    return ExchangeInfo(
        stamp=stamp,
        comments_start=comments_start,
        comments_end=comments_end,
        ctd_header_start=1,
        ctd_header_end=1,
        params=params,
        units=units,
        data_start=data_start,
        data_end=data_end,
        post_data_start=post_data_start,
        post_data_end=post_data_end,
    )


def _prepare_data_block(data_block: Tuple[str, ...]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(a.strip() for a in line.split(",")) for line in data_block)


def read_exchange(path):

    log.info("Trying to open as local filepath")
    with open(path, "rb") as f:
        data_raw = io.BytesIO(f.read())

    try:
        log.info("Decoding as UTF-8")
        data = data_raw.read().decode("utf8")
    except UnicodeDecodeError as error:
        raise ExchangeEncodingError from error

    log.info("Checking for BOM")
    if data.startswith("\ufeff"):
        raise ExchangeBOMError

    log.info("Checking Line Endings")
    if "\r" in data:
        raise ExchangeLEError

    log.info("Detecting file type")
    if data.startswith("BOTTLE"):
        ftype = FileType.BOTTLE
    elif data.startswith("CTD"):
        ftype = FileType.CTD
    else:
        raise ExchangeMagicNumberError

    log.info("Found filetype: %s", ftype.name)

    data_lines = tuple(data.splitlines())

    file_parts = _get_parts(data_lines, ftype=ftype)

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

    log.debug(whp_errors)

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

    log.debug("Preparting Data Block")
    data_block = _prepare_data_block(data_lines[file_parts.data])
    log.debug("%s ... %s", data_block[0], data_block[-1])

    log.debug("Extracting Exchange File Keys")

    expocode_col = whp_params[ExchangeCompositeKey.EXPOCODE]
    stnnbr_col = whp_params[ExchangeCompositeKey.STNNBR]
    castno_col = whp_params[ExchangeCompositeKey.CASTNO]
    sampno_col = whp_params[ExchangeCompositeKey.SAMPNO]

    exchange_keys = []
    for idx, line in enumerate(data_block):
        expocode = line[expocode_col]
        stnnbr = line[stnnbr_col]
        castno = line[castno_col]
        sampno = line[sampno_col]

        exchange_keys.append(
            (
                ExchangeCompositeKey(
                    expocode=ExchangeCompositeKey.EXPOCODE.data_type(expocode),
                    station=ExchangeCompositeKey.STNNBR.data_type(stnnbr),
                    cast=ExchangeCompositeKey.CASTNO.data_type(castno),
                    sample=ExchangeCompositeKey.SAMPNO.data_type(sampno),
                ),
                idx,
            )
        )

    log.debug("Extracting XYZT Coordinates")
    arr = np.array(data_block)
    log.debug(arr[:, whp_params[ExchangeCompositeKey.CASTNO]].astype("float16"))
