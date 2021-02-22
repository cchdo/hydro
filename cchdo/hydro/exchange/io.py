from __future__ import annotations
from collections import deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import (
    Union,
    Iterable,
    Tuple,
    Optional,
    Callable,
    Dict,
    List,
)
import io
from zipfile import is_zipfile, ZipFile
from operator import itemgetter
from datetime import datetime
from warnings import warn
import logging

import requests

from .containers import (
    IntermediateDataPoint,
    Exchange,
    FileType,
    ExchangeCompositeKey,
    ExchangeXYZT,
    ExchangeDataPoint,
)
from cchdo.params import WHPNames, WHPName
from .exceptions import (
    ExchangeEncodingError,
    ExchangeBOMError,
    ExchangeLEError,
    ExchangeMagicNumberError,
    ExchangeEndDataError,
    ExchangeDuplicateParameterError,
    ExchangeOrphanFlagError,
    ExchangeFlaglessParameterError,
    ExchangeParameterUnitAlignmentError,
    ExchangeDataColumnAlignmentError,
    ExchangeParameterUndefError,
    ExchangeFlagUnitError,
    ExchangeDataPartialKeyError,
    ExchangeDuplicateKeyError,
    ExchangeDataPartialCoordinateError,
    ExchangeRecursiveZip,
)

from .merge import merge_ex

log = logging.getLogger(__name__)

# WHPNameIndex represents a Name to Column index in an exchange file
WHPNameIndex = Dict[WHPName, int]
# WHPParamUnit represents the paired up contents of the Parameter and Unit lines
# in an exchange file
WHPParamUnit = Tuple[str, Optional[str]]


def _extract_comments(data: deque, include_post_content: bool = True) -> str:
    """Destructively extract the comments from exchange data.

    Exchange files may have zero or more lines of meaningless (to the format) comments.
    Between the ``CTD`` or ``BOTTLE`` stamp line and the start of the meaningful content of the file. These must be prefixed with a ``#`` character.
    Optionally, there might also be any amount of content after the ``END_DATA`` line of an exchange file.
    By default this function will extract that as well and append it to the retried comment string.
    This function will remove all the leading "#" from the comments.

    .. warning::

       This function expects the "stamp" line to have been popped from the deque already.

    :param collections.deque data: A deque containing the separated lines of an exchange file with the first line already popped.
    :param bool include_post_content: If True, include any post ``END_DATA`` content as part of the comments, default: True.
    :return: The extracted comment lines from the deque
    :rtype: str
    """
    comments = []
    while data[0].startswith("#"):
        comments.append(data.popleft().lstrip("#"))

    if include_post_content:
        post_content = []
        while data[-1] != "END_DATA":
            post_content.append(data.pop())
        comments.extend(reversed(post_content))

    return "\n".join(comments)


def _bottle_get_params(params_units: Iterable[WHPParamUnit]) -> WHPNameIndex:
    """Given an ordered iterable of param, unit pairs, return the index of the column in the datafile for known WHP params.

    Exchange files have comma separated parameter names on one line, and the corresponding units on the next.
    This function will search for this name+unit pair in the builtin database of known WHP parameter names and return a mapping of :py:class:`~hydro.data.WHPName` to column indicies.

    It is currently an error for the parameter in a file to not be in the built in database.

    This function will ignore uncertainty (error) columns and flag columns, those are parsed by other functions.

    .. warning::

        Convert semantically empty units (e.g. empty string, all whitespace) to None before passing into this function

    .. note::

        The parameter name database will convert unambiguous aliases to their canonical exchange parameter and unit pair.

    :param params_units: Paired (e.g. zip) parameter names and units
    :type params_units: tuple in the form (str, str) or (str, None)
    :returns: Mapping of :py:class:`~hydro.data.WHPName` to column indicies
    :rtype: dict with keys of :py:class:`~hydro.data.WHPName` and values of int
    :raises ExchangeParameterUndefError: if the parameter unit pair cannot be found in the built in database
    """
    params = {}
    for index, (param, unit) in enumerate(params_units):
        if param in WHPNames.error_cols:
            continue
        if param.endswith("_FLAG_W"):
            continue
        try:
            params[WHPNames[(param, unit)]] = index
        except KeyError as error:
            raise ExchangeParameterUndefError(
                f"missing parameter def {(param, unit)}"
            ) from error
    return params


def _bottle_get_flags(
    params_units: Iterable[WHPParamUnit], whp_params: WHPNameIndex
) -> WHPNameIndex:
    """Given an ordered iterable of param unit pairs and WHPNames known to be in the file, return the index of the column indicies of the flags for the WHPNames.

    Exchange files can have status flags for some of the parameters.
    Flag columns must have no units.
    Some parameters must not have status flags, these include the spatiotemporal parameters (e.g. lat, lon, but also pressure) and the sample identifying parameters (expocode, station, cast, sample, but *not* bottle id).

    :param params_units: Paired (e.g. zip) parameter names and units
    :type params_units: tuple in the form (str, str) or (str, None)
    :param whp_params: Mapping of parameters known to be in the file, this is the output of :py:func:`._bottle_get_params`
    :type whp_params: Mapping of :py:class:`~hydro.data.WHPName` to int
    :returns: Mapping of :py:class:`~hydro.data.WHPName` to column indicies for the status flag column
    :rtype: dict with keys of :py:class:`~hydro.data.WHPName` and values of int
    :raises ExchangeFlagUnitError: if the flag column has units other than None
    :raises ExchangeFlaglessParameterError: if the flag column is for a parameter not allowed to have status flags
    :raises ExchangeOrphanFlagError: if the flag column is for a parameter not in the passed in mapping of whp_params
    """
    param_flags = {}
    whp_params_names = {x.whp_name: x for x in whp_params.keys()}

    for index, (param, unit) in enumerate(params_units):
        if not param.endswith("_FLAG_W"):
            continue

        if unit is not None:
            raise ExchangeFlagUnitError

        data_col = param.replace("_FLAG_W", "")
        try:
            whpname = whp_params_names[data_col]
            if whpname.flag_w is None:
                raise ExchangeFlaglessParameterError(f"{data_col}")
            param_flags[whpname] = index
        except KeyError as error:
            # we might have an alias...
            for name in whp_params:
                potential = [k[0] for k, v in WHPNames.items() if v == name]
                if data_col in potential:
                    param_flags[name] = index
                    break
            else:
                raise ExchangeOrphanFlagError(f"{data_col}") from error

    return param_flags


def _bottle_get_errors(
    params_units: Iterable[WHPParamUnit], whp_params: WHPNameIndex
) -> WHPNameIndex:
    """Given an ordered iterable of param unit pairs and WHPNames known to be in the file, return the index of the column indicies of the errors/uncertanties for the WHPNames.

    Some parameters may have uncertanties associated with them, this function finds those columns and pairs them with the correct parameter.

    .. note::

        There is no programable way to find the error columns for a given unit (e.g. no common suffix like the flags).
        This must be done via lookup in the built in database of params.

    :param params_units: Paired (e.g. :py:func:`zip`) parameter names and units
    :type params_units: tuple in the form (str, str) or (str, None)
    :param whp_params: Mapping of parameters known to be in the file, this is the output of :py:func:`._bottle_get_params`
    :type whp_params: Mapping of :py:class:`~hydro.data.WHPName` to int
    :returns: Mapping of :py:class:`~hydro.data.WHPName` to column indicies for the error column
    :rtype: dict with keys of :py:class:`~hydro.data.WHPName` and values of int
    """
    param_errs = {}

    for index, (param, _unit) in enumerate(params_units):
        if param not in WHPNames.error_cols:
            continue

        for name in whp_params.keys():
            if name.error_name == param:
                param_errs[name] = index

    return param_errs


def _bottle_line_parser(
    names_index: WHPNameIndex, flags_index: WHPNameIndex, errors_index: WHPNameIndex
) -> Callable[[str], Dict[WHPName, IntermediateDataPoint]]:
    def _none_factory_factory(*args, **kwargs):
        def _none_factory(*args, **kwargs):
            return None

        return _none_factory

    GetterType = Dict[WHPName, Callable[[List[str]], Optional[str]]]

    data_getters = {}
    flag_getters: GetterType = defaultdict(_none_factory_factory)
    error_getters: GetterType = defaultdict(_none_factory_factory)

    for name, data_col in names_index.items():
        data_getters[name] = itemgetter(data_col)

        if flag_col := flags_index.get(name):
            flag_getters[name] = itemgetter(flag_col)

        if error_col := errors_index.get(name):
            error_getters[name] = itemgetter(error_col)

    def line_parser(line: str) -> Dict[WHPName, IntermediateDataPoint]:
        split_line = [s.strip() for s in line.split(",")]
        parsed = {}
        for name in names_index:
            data: str = data_getters[name](split_line)
            flag: Optional[str] = flag_getters[name](split_line)
            error: Optional[str] = error_getters[name](split_line)

            parsed[name] = IntermediateDataPoint(data, flag, error)

        return parsed

    return line_parser


def _bottle_merge_sample_times(date, time):
    if date.startswith("-999") != time.startswith("-999"):
        raise ValueError("BTL_TIME or BTL_DATE have mismatched fill values")
    if date.startswith("-999"):
        return None
    if len(time) != 4:
        warn(f"BTL_TIME is not zero padded: {time}")
        time = time.zfill(4)
    return datetime.strptime(f"{date}{time}", "%Y%m%d%H%M")


def _ctd_get_header(line, dtype=str):
    header, value = (part.strip() for part in line.split("="))
    return header, dtype(value)


def read_exchange(
    filename_or_obj: Union[str, Path, io.BufferedIOBase],
    parallelize="processpool",
    recursed=False,
) -> Exchange:
    """Open an exchange file and return an :class:`hydro.exchange.Exchange` object"""

    if isinstance(filename_or_obj, str) and filename_or_obj.startswith("http"):
        log.info("Loading object over http")
        data_raw = io.BytesIO(requests.get(filename_or_obj).content)

    elif isinstance(filename_or_obj, (str, Path)):
        log.info("Loading object from local file path")
        with open(filename_or_obj, "rb") as f:
            data_raw = io.BytesIO(f.read())

    elif isinstance(filename_or_obj, io.BufferedIOBase):
        log.info("Loading object open file object")
        data_raw = io.BytesIO(filename_or_obj.read())

    if is_zipfile(data_raw):
        if recursed is True:
            raise ExchangeRecursiveZip

        recursed_read_exchange = partial(read_exchange, recursed=True)

        data_raw.seek(0)  # is_zipfile moves the "tell" position
        zip_contents = []
        with ZipFile(data_raw) as zf:
            for zipinfo in zf.infolist():
                zip_contents.append(io.BytesIO(zf.read(zipinfo)))

        if parallelize == "processpool":
            with ProcessPoolExecutor() as executor:
                results = executor.map(recursed_read_exchange, zip_contents)
        else:
            results = map(recursed_read_exchange, zip_contents)

        return merge_ex(*results)

    data_raw.seek(0)  # is_zipfile moves the "tell" position

    try:
        data = data_raw.read().decode("utf8")
    except UnicodeDecodeError as error:
        raise ExchangeEncodingError from error

    if data.startswith("\ufeff"):
        raise ExchangeBOMError

    if "\r" in data:
        raise ExchangeLEError

    if data.startswith("BOTTLE"):
        ftype = FileType.BOTTLE
    elif data.startswith("CTD"):
        ftype = FileType.CTD
    else:
        raise ExchangeMagicNumberError

    log.info(f"Found filetype: {ftype.name}")

    data_lines = deque(data.splitlines())
    stamp = data_lines.popleft()

    if "END_DATA" not in data_lines:
        raise ExchangeEndDataError

    comments = "\n".join([stamp, _extract_comments(data_lines)])

    # Strip end_data
    data_lines.remove("END_DATA")

    # CTD Divergence
    ctd_params = []
    ctd_units: List[None] = []
    ctd_values = []
    if ftype == FileType.CTD:
        param, value = _ctd_get_header(data_lines.popleft(), dtype=int)
        if param != "NUMBER_HEADERS":
            raise ValueError()
        number_headers = value

        for _ in range(number_headers - 1):
            param, value = _ctd_get_header(data_lines.popleft())
            ctd_params.append(param)
            ctd_units.append(None)
            ctd_values.append(value)

    params = [param.strip() for param in data_lines.popleft().split(",")]
    # we can have a bunch of empty strings as units, we want these to be
    # None to match what would be in a WHPName object
    units = [
        x if x != "" else None
        for x in [unit.strip() for unit in data_lines.popleft().split(",")]
    ]

    # at this point the data_lines should ONLY contain data/flags

    if ftype == FileType.CTD:
        params = [*ctd_params, *params]
        units = [*ctd_units, *units]

    # column labels must be unique
    if len(params) != len(set(params)):
        raise ExchangeDuplicateParameterError

    # In initial testing, it was discovered that approx half the ctd files
    # had trailing commas in just the params and units lines
    if params[-1] == "" and units[-1] is None:
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

    # ensure we will read the ENTIRE file
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

    line_parser = _bottle_line_parser(whp_params, whp_flags, whp_errors)

    exchange_data: Dict[ExchangeCompositeKey, dict] = {}
    coordinates = {}

    for data_line in data_lines:
        cols = [x.strip() for x in data_line.split(",")]

        if ftype == FileType.CTD:
            cols = [*ctd_values, *cols]
            data_line = f"{','.join(ctd_values)},{data_line}"

        if len(cols) != column_count:
            raise ExchangeDataColumnAlignmentError
        parsed_data_line = line_parser(data_line)

        if ftype == FileType.CTD:
            parsed_data_line[ExchangeCompositeKey.SAMPNO] = parsed_data_line[
                ExchangeXYZT.CTDPRS
            ]

        try:
            key = ExchangeCompositeKey.from_data_line(parsed_data_line)
        except KeyError as error:
            raise ExchangeDataPartialKeyError from error

        if key in exchange_data:
            raise ExchangeDuplicateKeyError(f"{key}")

        try:
            coord = ExchangeXYZT.from_data_line(parsed_data_line)
        except KeyError as error:
            raise ExchangeDataPartialCoordinateError from error

        row_data = {
            param: ExchangeDataPoint.from_ir(param, ir)
            for param, ir in parsed_data_line.items()
        }

        exchange_data[key] = dict(filter(lambda di: di[1].flag != 9, row_data.items()))
        coordinates[key] = coord

    return Exchange(
        file_type=ftype,
        comments=comments,
        parameters=tuple(whp_params.keys()),
        flags=tuple(whp_flags.keys()),
        errors=tuple(whp_errors.keys()),
        keys=tuple(exchange_data.keys()),
        coordinates=coordinates,
        data=exchange_data,
    )
