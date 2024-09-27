import json
import warnings
from importlib.resources import read_text
from io import BytesIO
from zipfile import ZipFile

import pytest
import xarray as xr
from xarray.testing import assert_identical

from cchdo.hydro.accessors import CCHDOAccessor
from cchdo.hydro.exchange import FileType, read_csv, read_exchange

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


def test_coards_no_comments():
    # note that the differet bottle and sampno are intentional
    test_data = b"""BOTTLE,test
# some comment
EXPOCODE,STNNBR,CASTNO,SAMPNO,BTLNBR,DATE,TIME,LATITUDE,LONGITUDE,CTDPRS
,,,,,,,,,DBAR
TEST          ,1       ,  1,1          ,2          ,20200101,0000,        0,        0,        0
END_DATA
"""
    ds = read_exchange(test_data)
    ds.attrs["comments"] = ""
    ds.cchdo.to_coards()


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


def test_nc_serialize_all_ctd(tmp_path):
    """A crash was discovered when the ctd elapsed time param was present, and was seralized to disk then read back in"""
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
    nc = tmp_path / "test.nc"
    ds.to_netcdf(nc)
    ds = xr.load_dataset(nc)
    # the magic slice removes the stamp and the newline with #
    ds.cchdo.to_exchange()
    ds.cchdo.to_coards()
    ds.cchdo.to_woce()
    ds.cchdo.to_sum()


def test_nc_serialize_all_ctdetime(tmp_path):
    """A crash was discovered when the ctd elapsed time param was present, and was seralized to disk then read back in"""
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
CTDPRS,CTDETIME
DBAR,SECONDS
0,-999
END_DATA
"""
    ds = read_exchange(test_data)
    nc = tmp_path / "test.nc"
    ds.to_netcdf(nc)
    ds = xr.load_dataset(nc)
    # the magic slice removes the stamp and the newline with #
    ds.cchdo.to_exchange()
    ds.cchdo.to_coards()
    ds.cchdo.to_woce()
    ds.cchdo.to_sum()


def test_woce_ctd_no_flags(tmp_path):
    """make sure data is written to the woce files when there are no qc flags"""
    test_data = b"""EXPOCODE,SECT_ID,STNNBR,CASTNO,DATE,TIME,LATITUDE,LONGITUDE,CTDPRS [DBAR],CTDTMP [DEG C],CTDSAL [PSS-78],CTDOXY [UMOL/KG],CTDFLUOR [MG/M^3],CTDBEAMCP [/METER]
64PE20110724,AR07E,9,1,20110729,1919,59.57017,-38.77183,3006.0,1.2096,34.8913,304.0,0.0,0.161
"""
    ds = read_csv(test_data, ftype="C")
    zip_data = BytesIO(ds.cchdo.to_woce())
    with ZipFile(zip_data) as zf:
        fname = zf.namelist()[0]
        with zf.open(fname) as ctd:
            # this asserts that the above data block exists somewhere in the resulting file
            # the previous bug would just have a blank data block section
            assert b"3006.0  1.2096 34.8913   304.0" in ctd.read()


def test_coards_ctdnobs_with_missing():
    """Test a condition where exception might be thrown when a float array with nans is cast to int

    This is very platform dependent and depends on undefined behavior in C. Lucky for us we can see if
    numpy is issuing a runtime warning (they are considering makeing this throw in the future anyway)
    """
    test_data = b"""EXPOCODE,SECT_ID,STNNBR,CASTNO,DATE,TIME,LATITUDE,LONGITUDE,CTDPRS [DBAR],CTDTMP [DEG C],CTDSAL [PSS-78],CTDNOBS
64PE20110724,AR07E,9,1,20110729,1919,59.57017,-38.77183,3006.0,1.2096,34.8913,-999
"""
    ds = read_csv(test_data, ftype="C")
    with warnings.catch_warnings(record=True) as w:
        ds.cchdo.to_coards()
        # simply assert that no warnings were issued durring the test
        # numpy would issue a RuntimeWarning if an unsafe cast occured
        assert len(w) == 0
