from importlib.resources import read_text
import json

import pytest

from hydro.accessors import MiscAccessor
from hydro.exchange import FileType

exp_stn_cast = json.loads(read_text("cchdo.hydro.tests.data", "stns_test_data.json"))


@pytest.mark.parametrize("expocode", ["318M20130321", "320620140320", "33KI159/1"])
@pytest.mark.parametrize("station", exp_stn_cast["stations"])
@pytest.mark.parametrize("cast", [0, 1, 10, 100, 999, 9999])
@pytest.mark.parametrize("profile_type", [FileType.BOTTLE, FileType.CTD])
@pytest.mark.parametrize("profile_count", [1, 2])
@pytest.mark.parametrize("ftype", ["cf", "exchange"])
def test_gen_fname_machinery(
    expocode, station, cast, profile_type, profile_count, ftype
):
    MiscAccessor._gen_fname(expocode, station, cast, profile_type, profile_count, ftype)
