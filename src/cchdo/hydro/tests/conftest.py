import io
from importlib.resources import open_binary

import pytest

from cchdo.hydro.exchange import read_csv, read_exchange
from cchdo.hydro.exchange.helpers import simple_bottle_exchange


@pytest.fixture
def nc_empty():
    return read_exchange(io.BytesIO(simple_bottle_exchange()))


@pytest.fixture
def nc_placeholder():
    params = ("OXYGEN", "OXYGEN_FLAG_W", "DELC14", "DELC14_FLAG_W", "C14ERR")
    units = ("UMOL/KG", "", "/MILLE", "", "/MILLE")
    data = ("-999", "1", "-999", "1", "-999")
    return read_exchange(
        io.BytesIO(simple_bottle_exchange(params=params, units=units, data=data))
    )


@pytest.fixture(scope="function")
def nc_placeholders():
    # Multiple empty slots for this one
    with open_binary("cchdo.hydro.tests.data", "merge_placeholders.csv") as f:
        return read_csv(f)
