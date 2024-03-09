import json
from importlib.resources import read_text

import pytest
from xarray.testing import assert_identical

from ..accessors import CCHDOAccessor
from ..exchange import FileType, read_exchange

exp_stn_cast = json.loads(read_text("cchdo.hydro.tests.data", "stns_test_data.json"))


@pytest.mark.parametrize("expocode", ["318M20130321", "320620140320", "33KI159/1"])
@pytest.mark.parametrize("station", exp_stn_cast["stations"])
@pytest.mark.parametrize("cast", [0, 1, 10, 100, 999, 9999])
@pytest.mark.parametrize("profile_type", [FileType.BOTTLE, FileType.CTD])
@pytest.mark.parametrize("profile_count", [1, 2])
@pytest.mark.parametrize("ftype", ["cf", "exchange", "coards", "woce"])
def test_gen_fname_machinery(
    expocode, station, cast, profile_type, profile_count, ftype
):
    CCHDOAccessor._gen_fname(
        expocode, station, cast, profile_type, profile_count, ftype
    )


def test_exchange_bottle_round_trip():
    # note that the differet bottle and sampno are intentional
    test_data = b"""BOTTLE,test
# some comment
EXPOCODE,STNNBR,CASTNO,SAMPNO,BTLNBR,DATE,TIME,LATITUDE,LONGITUDE,CTDPRS
,,,,,,,,,DBAR
TEST          ,1       ,  1,1          ,2          ,20200101,0000,        0,        0,        0
END_DATA
"""
    ds = read_exchange(test_data)
    # the magic slice removes the stamp and the newline with #
    rt = read_exchange(ds.cchdo.to_exchange()[25:])
    assert_identical(ds, rt)


def test_exchange_bottle_round_trip_with_alt():
    # note that the differet bottle and sampno are intentional
    test_data = b"""BOTTLE,test
# some comment
EXPOCODE,STNNBR,CASTNO,SAMPNO,BTLNBR,DATE,TIME,LATITUDE,LONGITUDE,CTDPRS,CTDSAL,CTDSAL_ALT_1,CTDSAL_ALT_1_FLAG_W,SILCAT,SILCAT_ALT_1,SILUNC_ALT_1
,,,,,,,,,DBAR,PSS-78,PSS-78,,UMOL/KG,UMOL/KG,UMOL/KG
TEST          ,1       ,  1,1          ,2          ,20200101,0000,        0,        0,        0,35,36,2,10,10,3
END_DATA
"""
    ds = read_exchange(test_data)
    # the magic slice removes the stamp and the newline with #
    rt = read_exchange(ds.cchdo.to_exchange()[25:])
    assert_identical(ds, rt)


def test_exchange_bottle_round_trip_cdom():
    # note that the differet bottle and sampno are intentional
    # note that we do want the print precision of the two CDOMs to differ
    test_data = b"""BOTTLE,test
# some comment
EXPOCODE,STNNBR,CASTNO,SAMPNO,BTLNBR,DATE,TIME,LATITUDE,LONGITUDE,CTDPRS,CDOM-300,CDOM-300_FLAG_W,CDOM-325,CDOM-325_FLAG_W
,,,,,,,,,DBAR,/METER,,/METER,
TEST          ,1       ,  1,1          ,2          ,20200101,0000,        0,        0,        0,      0.052,2,     0.0091,2
END_DATA
"""
    ds = read_exchange(test_data)
    # the magic slice removes the stamp and the newline with #
    rt = read_exchange(ds.cchdo.to_exchange()[25:])
    assert_identical(ds, rt)


def test_exchange_ctd_round_trip():
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
    ds = read_exchange(test_data)
    # the magic slice removes the stamp and the newline with #
    rt = read_exchange(ds.cchdo.to_exchange()[22:])
    assert_identical(ds, rt)
