import io
from importlib.resources import open_binary

import numpy as np
import pytest

from cchdo.hydro import accessors  # noqa: F401
from cchdo.hydro.exchange import read_exchange
from cchdo.hydro.exchange.exceptions import (
    ExchangeBOMError,
    ExchangeDuplicateParameterError,
    ExchangeEncodingError,
    ExchangeParameterUndefError,
)
from cchdo.hydro.exchange.helpers import simple_bottle_exchange


def test_btl_date_time():
    raw = simple_bottle_exchange(
        params=("BTL_DATE", "BTL_TIME"), units=("", ""), data=("20200101", "1234")
    )
    ex_xr = read_exchange(io.BytesIO(raw))

    assert "bottle_time" in ex_xr.variables
    assert "bottle_date" not in ex_xr.variables
    assert ex_xr["bottle_time"].values == [[np.datetime64("2020-01-01T12:34")]]


def test_btl_date_time_fill_round_trip():
    raw = simple_bottle_exchange(
        params=("BTL_DATE", "BTL_TIME"), units=("", ""), data=("-999", "-999")
    )
    ex_xr = read_exchange(io.BytesIO(raw))

    assert "bottle_time" in ex_xr.variables
    np.testing.assert_array_equal(ex_xr["bottle_time"].values, [[np.datetime64("nat")]])

    # test the round trip
    ex = ex_xr.cchdo.to_exchange()
    assert b"nan" not in ex
    ex_xr_rt = read_exchange(io.BytesIO(ex))
    np.testing.assert_array_equal(
        ex_xr_rt["bottle_time"].values, [[np.datetime64("nat")]]
    )


def test_btl_date_time_missing_warn():
    raw = simple_bottle_exchange(
        params=("BTL_DATE", "BTL_TIME"), units=("", ""), data=("20200101", "34")
    )
    with pytest.warns(UserWarning):
        ex_xr = read_exchange(io.BytesIO(raw))

    assert ex_xr["bottle_time"].values == [[np.datetime64("2020-01-01T00:34")]]


def test_ctd_nan():
    test_data = b"""CTD,test
# some comment
NUMBER_HEADERS = 8
EXPOCODE = TEST
STNNBR = 1
CASTNO = 1
DATE = 20200101
TIME = 0000
LATITUDE = nan
LONGITUDE = 0
CTDPRS
DBAR
0
END_DATA
"""
    with pytest.raises(ValueError):
        read_exchange(test_data)


def test_file_seperator():
    test_data = b"""CTD,test
# some comment
NUMBER_HEADERS = 8
EXPOCODE = TEST
STNNBR = 1
CASTNO = 1
DATE = 20200101
TIME = 0000
LATITUDE = 0
LONGITUDE = 0
CTDPRS
DBAR
0
END_DATA
"""
    ex = read_exchange(test_data + test_data, file_seperator="END_DATA")
    assert ex.sizes["N_PROF"] == 2


@pytest.mark.parametrize(
    "data,error",
    [
        (io.BytesIO("À".encode("latin-1")), ExchangeEncodingError),
        (io.BytesIO("\ufeffBOTTLE".encode()), ExchangeBOMError),
    ],
)
def test_reject_bad_examples(data, error):
    with pytest.raises(error):
        read_exchange(data)


@pytest.mark.parametrize(
    "uri", ["https://cchdo.ucsd.edu/exchange.csv", "http://cchdo.ucsd.edu/exchange.csv"]
)
def test_http_loads(uri, requests_mock):
    requests_mock.get(uri, content=simple_bottle_exchange())
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


def test_duplicate_name_different_units_keep_flags():
    raw = simple_bottle_exchange(
        params=("CTDTMP", "CTDTMP_FLAG_W", "CTDTMP", "CTDTMP_FLAG_W"),
        units=("ITS-90", "", "IPTS-68", ""),
        data=("-999", "9", "-999", "9"),
    )
    ex_xr = read_exchange(io.BytesIO(raw))
    assert "ctd_temperature" in ex_xr.variables
    assert "ctd_temperature_68" in ex_xr.variables
    assert "ctd_temperature_qc" in ex_xr.variables
    assert "ctd_temperature_68_qc" in ex_xr.variables


def test_duplicate_name_same_units():
    raw = simple_bottle_exchange(
        params=("CTDTMP", "CTDTMP"), units=("ITS-90", "ITS-90"), data=("-999", "-999")
    )
    with pytest.raises(ExchangeDuplicateParameterError):
        read_exchange(io.BytesIO(raw))


def test_multiple_unknown_params():
    raw = simple_bottle_exchange(
        params=("TEST1", "TEST2"), units=("TEST3", "TEST4"), data=("-999", "-999")
    )
    with pytest.raises(ExchangeParameterUndefError) as execinfo:
        read_exchange(io.BytesIO(raw))

    assert hasattr(execinfo.value, "error_data")
    assert execinfo.value.error_data == ["TEST1 [TEST3]", "TEST2 [TEST4]"]


def test_multiple_unknown_params_ignore():
    raw = simple_bottle_exchange(
        params=("TEST1", "TEST2"), units=("TEST3", "TEST4"), data=("-999", "-999")
    )
    read_exchange(io.BytesIO(raw), ignore_columns=("TEST1 [TEST3]", "TEST2 [TEST4]"))


def test_multiple_unknown_params_flag():
    raw = simple_bottle_exchange(
        params=("OXYGEN", "OXYGEN_FLAG_W"), units=("UMOL/KG", ""), data=("-999", "2")
    )
    # The flag input above is intentionally invalid, this should ignore that
    ex = read_exchange(io.BytesIO(raw), ignore_columns=("OXYGEN [UMOL/KG]_FLAG_W",))
    assert "oxygen_qc" not in ex


def test_alternate_params():
    raw = simple_bottle_exchange(
        params=("CTDTMP", "CTDTMP_ALT_1"),
        units=("ITS-90", "ITS-90"),
        data=("-999", "-999"),
    )
    ex_xr = read_exchange(io.BytesIO(raw))
    assert "ctd_temperature" in ex_xr.variables
    assert "ctd_temperature_alt_1" in ex_xr.variables


def test_alternate_params_flags():
    raw = simple_bottle_exchange(
        params=("CTDTMP", "CTDTMP_ALT_1", "CTDTMP_ALT_1_FLAG_W"),
        units=("ITS-90", "ITS-90", ""),
        data=("-999", "-999", "9"),
    )
    ex_xr = read_exchange(io.BytesIO(raw))
    assert "ctd_temperature" in ex_xr.variables
    assert "ctd_temperature_alt_1" in ex_xr.variables
    assert "ctd_temperature_alt_1_qc" in ex_xr.variables
    assert "ctd_temperature_qc" not in ex_xr.variables


def test_fix_bottle_time_span():
    test_data = open_binary("cchdo.hydro.tests.data", "btl_time_span.csv")
    expected = np.array(
        [
            [
                "2012-09-05T01:15:00.000000000",
                "2012-09-05T01:33:00.000000000",
                "2012-09-05T01:32:00.000000000",
                "2012-09-05T01:28:00.000000000",
                "2012-09-05T01:27:00.000000000",
                "2012-09-05T01:26:00.000000000",
                "2012-09-05T01:25:00.000000000",
                "2012-09-05T01:23:00.000000000",
                "2012-09-05T01:22:00.000000000",
                "2012-09-05T01:20:00.000000000",
                "2012-09-05T01:18:00.000000000",
                "2012-09-05T01:16:00.000000000",
                "2012-09-05T01:13:00.000000000",
                "2012-09-05T01:10:00.000000000",
                "2012-09-05T01:08:00.000000000",
                "2012-09-05T01:05:00.000000000",
                "2012-09-05T01:02:00.000000000",
                "2012-09-05T00:59:00.000000000",
                "2012-09-05T00:55:00.000000000",
                "2012-09-05T00:50:00.000000000",
                "2012-09-05T00:45:00.000000000",
                "2012-09-05T00:41:00.000000000",
                "2012-09-05T00:36:00.000000000",
                "2012-09-05T00:31:00.000000000",
                "2012-09-05T00:26:00.000000000",
                "2012-09-05T00:20:00.000000000",
                "2012-09-05T00:15:00.000000000",
                "2012-09-05T00:15:00.000000000",
                "2012-09-05T00:10:00.000000000",
                "2012-09-05T00:04:00.000000000",
                "2012-09-05T00:04:00.000000000",
                "2012-09-04T23:59:00.000000000",
                "2012-09-04T23:54:00.000000000",
                "2012-09-04T23:54:00.000000000",
                "2012-09-04T23:48:00.000000000",
                "2012-09-04T23:43:00.000000000",
                "2012-09-04T23:41:00.000000000",
            ]
        ],
        dtype="datetime64[ns]",
    )

    ex = read_exchange(test_data)

    np.testing.assert_array_equal(ex.bottle_time.values, expected)


def test_empty_string_data():
    """A bug was found in new numpy where an all fill value string param would not convert succesfully"""
    raw = simple_bottle_exchange(params=("HPLC",), units=("",), data=("-999",))
    ex_xr = read_exchange(io.BytesIO(raw))

    assert "hplc_placeholder" in ex_xr.variables
    assert ex_xr.hplc_placeholder.values == [[""]]


def test_alternate_encoding():
    test_str = "Héllø Wörld"
    raw = simple_bottle_exchange(comments=test_str).decode("utf8").encode("latin1")
    with pytest.raises(ExchangeEncodingError):
        ex_xr = read_exchange(io.BytesIO(raw))

    with pytest.raises(ExchangeEncodingError):
        ex_xr = read_exchange(io.BytesIO(raw), encoding="shift-jis")

    ex_xr = read_exchange(io.BytesIO(raw), encoding="latin1")

    assert ex_xr.comments == f"BOTTLE,test\n{test_str}"
