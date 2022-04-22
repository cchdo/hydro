import io
import pytest

import numpy as np

from hydro.exchange import read_exchange

from hydro.exchange.exceptions import (
    ExchangeLEError,
    ExchangeBOMError,
    ExchangeEncodingError,
)

from hydro.exchange.helpers import simple_bottle_exchange


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


@pytest.mark.parametrize("flag", ["1", "2", "3", "4", "6", "7"])
def test_pressure_flags(flag):
    raw = simple_bottle_exchange(params=("CTDPRS_FLAG_W",), units=("",), data=(flag,))
    ex_xr = read_exchange(io.BytesIO(raw))

    assert "pressure_qc" in ex_xr.variables
    assert "pressure_qc" in ex_xr.pressure.attrs["ancillary_variables"]
    assert ex_xr.pressure_qc.data == [int(flag)]
    assert "pressure" in ex_xr.coords


@pytest.mark.parametrize("flag", ["5", "8", "9"])
def test_pressure_flags_bad(flag):
    raw = simple_bottle_exchange(params=("CTDPRS_FLAG_W",), units=("",), data=(flag,))
    with pytest.raises(ValueError):
        read_exchange(io.BytesIO(raw))


def test_duplicate_name_different_units():
    raw = simple_bottle_exchange(
        params=("CTDTMP", "CTDTMP"), units=("ITS-90", "IPTS-68"), data=("-999", "-999")
    )
    ex_xr = read_exchange(io.BytesIO(raw))
    assert "ctd_temperature" in ex_xr.variables
    assert "ctd_temperature_68" in ex_xr.variables


def test_duplicate_name_same_units():
    raw = simple_bottle_exchange(
        params=("CTDTMP", "CTDTMP"), units=("ITS-90", "ITS-90"), data=("-999", "-999")
    )
    with pytest.raises(ValueError):
        read_exchange(io.BytesIO(raw))
