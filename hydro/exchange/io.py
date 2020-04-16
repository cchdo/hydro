from __future__ import annotations
from collections import deque, defaultdict
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
from zipfile import is_zipfile
from operator import itemgetter

import requests

from .containers import (
    IntermediateDataPoint,
    Exchange,
    FileType,
    ExchangeCompositeKey,
    ExchangeXYZT,
    ExchangeDataPoint,
)
from ..data import WHPNames, WHPName
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
)

WHPNameIndex = Dict[WHPName, int]


def _extract_comments(data: deque, include_post_content: bool = True) -> str:
    comments = []
    while data[0].startswith("#"):
        comments.append(data.popleft().lstrip("#"))

    if include_post_content:
        post_content = []
        while data[-1] != "END_DATA":
            post_content.append(data.pop())
        comments.extend(reversed(post_content))

    return "\n".join(comments)


def _bottle_get_params(
    params_units: Iterable[Tuple[str, Optional[str]]]
) -> WHPNameIndex:
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
    params_units: Iterable[Tuple[str, Optional[str]]], whp_params: WHPNameIndex
) -> WHPNameIndex:

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
    params_units: Iterable[Tuple[str, Optional[str]]], whp_params: WHPNameIndex
) -> WHPNameIndex:
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


def read_exchange(filename_or_obj: Union[str, Path, io.BufferedIOBase]) -> Exchange:
    """Open an exchange file and return an :class:`hydro.exchange.Exchange` object
    """

    if isinstance(filename_or_obj, str) and filename_or_obj.startswith("http"):
        data_raw = io.BytesIO(requests.get(filename_or_obj).content)

    elif isinstance(filename_or_obj, (str, Path)):
        with open(filename_or_obj, "rb") as f:
            data_raw = io.BytesIO(f.read())

    elif isinstance(filename_or_obj, io.BufferedIOBase):
        data_raw = io.BytesIO(filename_or_obj.read())

    if is_zipfile(data_raw):
        raise NotImplementedError("zip files not supported yet")
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

    data_lines = deque(data.splitlines())
    stamp = data_lines.popleft()

    if "END_DATA" not in data_lines:
        raise ExchangeEndDataError

    comments = "\n".join([stamp, _extract_comments(data_lines)])

    # Strip end_data
    data_lines.remove("END_DATA")

    params = data_lines.popleft().split(",")
    # we can have a bunch of empty strings as units, we want these to be
    # None to match what would be in a WHPName object
    units = [x if x != "" else None for x in data_lines.popleft().split(",")]

    # at this point the data_lines should ONLY contain data/flags

    # column labels must be unique
    if len(params) != len(set(params)):
        raise ExchangeDuplicateParameterError

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
        if len(cols) != column_count:
            raise ExchangeDataColumnAlignmentError
        parsed_data_line = line_parser(data_line)
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
