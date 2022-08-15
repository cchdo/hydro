import io

import pytest

from ..exchange.helpers import simple_bottle_exchange
from ..exchange import read_exchange


@pytest.fixture
def nc_empty():
    return read_exchange(io.BytesIO(simple_bottle_exchange()))


@pytest.fixture
def nc_placeholder():
    params = ("OXYGEN", "OXYGEN_FLAG_W")
    units = ("UMOL/KG", "")
    data = ("-999", "1")
    return read_exchange(
        io.BytesIO(simple_bottle_exchange(params=params, units=units, data=data))
    )
