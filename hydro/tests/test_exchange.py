import io
import pytest

from hydro.exchange import read_exchange, InvalidExchangeFileError


@pytest.mark.parametrize(
    "data,msg",
    [
        (io.BytesIO("À".encode("latin-1")), "utf8"),
        (io.BytesIO("\ufeffBOTTLE".encode("utf8")), "byte order mark"),
        (io.BytesIO("BOTTLE\r".encode("utf8")), "LF line endings"),
        (io.BytesIO("BOTTLE\r\n".encode("utf8")), "LF line endings"),
    ],
)
def test_reject_bad_examples(data, msg):
    with pytest.raises(InvalidExchangeFileError) as excinfo:
        read_exchange(data)
    assert msg in str(excinfo.value)


@pytest.mark.parametrize(
    "uri", ["https://cchdo.ucsd.edu/exchange.csv", "http://cchdo.ucsd.edu/exchange.csv"]
)
def test_http_loads(uri, requests_mock):
    requests_mock.get(uri, content="BOTTLE\r".encode("utf8"))
    with pytest.raises(InvalidExchangeFileError) as excinfo:
        read_exchange(uri)
    assert "LF line endings" in str(excinfo.value)
