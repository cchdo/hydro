import io
import pytest

import numpy as np

from hydro.exchange import read_exchange

from hydro.exchange.exceptions import (
    ExchangeLEError,
    ExchangeBOMError,
    ExchangeEncodingError,
)


def simple_bottle_exchange(params=None, units=None, data=None, comments: str = None):
    stamp = "BOTTLE,test"
    min_params = [
        "EXPOCODE",
        "STNNBR",
        "CASTNO",
        "SAMPNO",
        "LATITUDE",
        "LONGITUDE",
        "DATE",
        "TIME",
        "CTDPRS",
    ]
    min_units = ["", "", "", "", "", "", "", "", "DBAR"]
    min_line = ["TEST", "1", "1", "1", "0", "0", "20200101", "0000", "0"]
    end = "END_DATA"

    if params is not None:
        min_params.extend(params)
    if units is not None:
        min_units.extend(units)
    if data is not None:
        min_line.extend(data)

    if comments is not None:
        comments = "\n".join([f"#{line}" for line in comments.splitlines()])
        simple = "\n".join(
            [
                stamp,
                comments,
                ",".join(min_params),
                ",".join(min_units),
                ",".join(min_line),
                end,
            ]
        )
    else:
        simple = "\n".join(
            [stamp, ",".join(min_params), ",".join(min_units), ",".join(min_line), end]
        )
    return simple.encode("utf8")


def test_btl_date_time():
    raw = simple_bottle_exchange(
        params=("BTL_DATE", "BTL_TIME"), units=("", ""), data=("20200101", "1234")
    )
    ex_xr = read_exchange(io.BytesIO(raw))

    assert "bottle_time" in ex_xr.variables
    assert "bottle_date" not in ex_xr.variables
    assert ex_xr["bottle_time"].values == [[np.datetime64("2020-01-01T12:34")]]


def test_btl_date_time_missing_part():
    raw = simple_bottle_exchange(params=("BTL_DATE",), units=("",), data=("20200101",))
    with pytest.raises(ValueError):
        read_exchange(io.BytesIO(raw))
    raw = simple_bottle_exchange(params=("BTL_TIME",), units=("",), data=("0012",))
    with pytest.raises(ValueError):
        read_exchange(io.BytesIO(raw))


def test_btl_date_time_missing_warn():
    raw = simple_bottle_exchange(
        params=("BTL_DATE", "BTL_TIME"), units=("", ""), data=("20200101", "34")
    )
    with pytest.warns(UserWarning):
        ex_xr = read_exchange(io.BytesIO(raw))

    assert ex_xr["bottle_time"].values == [[np.datetime64("2020-01-01T00:34")]]


@pytest.mark.parametrize(
    "data,error",
    [
        (io.BytesIO("À".encode("latin-1")), ExchangeEncodingError),
        (io.BytesIO("\ufeffBOTTLE".encode("utf8")), ExchangeBOMError),
        (io.BytesIO("BOTTLE\r".encode("utf8")), ExchangeLEError),
        (io.BytesIO("BOTTLE\r\n".encode("utf8")), ExchangeLEError),
    ],
)
def test_reject_bad_examples(data, error):
    with pytest.raises(error):
        read_exchange(data)


@pytest.mark.parametrize(
    "uri", ["https://cchdo.ucsd.edu/exchange.csv", "http://cchdo.ucsd.edu/exchange.csv"]
)
def test_http_loads(uri, requests_mock):
    requests_mock.get(uri, content="BOTTLE\r".encode("utf8"))
    with pytest.raises(ExchangeLEError):
        read_exchange(uri)
