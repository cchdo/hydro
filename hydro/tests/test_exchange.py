import io
import pytest

from hydro.exchange import read_exchange
from hydro.exchange.exceptions import (
    ExchangeLEError,
    ExchangeBOMError,
    ExchangeEncodingError,
)


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
