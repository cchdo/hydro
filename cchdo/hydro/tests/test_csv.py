from importlib.resources import open_binary
from io import BytesIO

from ..exchange import read_csv


def test_read_csv():
    with open_binary("cchdo.hydro.tests.data", "btl_csv.csv") as f:
        test_data = f.read()

    read_csv(test_data)


def test_all_flags_kept():
    test_data = BytesIO(
        b"""EXPOCODE,STNNBR,CASTNO,SAMPNO,LATITUDE,LONGITUDE,DATE,TIME,CTDPRS [DBAR],CTDOXY [UMOL/KG],CTDOXY [UMOL/KG]_FLAG_W,CTDOXY [UMOL/L],CTDOXY [UMOL/L]_FLAG_W
TEST,1,1,1,0,0,20220101,0000,0,0,2,0,3"""
    )
    ds = read_csv(test_data)

    assert "pressure" in ds
    assert "ctd_oxygen" in ds
    assert "ctd_oxygen_qc" in ds
    assert "ctd_oxygen_umol_l" in ds
    assert "ctd_oxygen_umol_l_qc" in ds
    assert ds.ctd_oxygen_qc.to_numpy() == [[2]]
    assert ds.ctd_oxygen_umol_l_qc.to_numpy() == [[3]]


def test_all_error_params():
    """Tests a condition where the presence of an error param was causing other params to be invalid (BTL DATE and TIME)

    Just needs to read without crashing
    """
    test_data = BytesIO(
        b"""EXPOCODE,STNNBR,CASTNO,SAMPNO,LATITUDE,LONGITUDE,DATE,TIME,CTDPRS [DBAR],TRITUM [KBQ/M^3],TRITUM [KBQ/M^3]_FLAG_W,TRITER [KBQ/M^3],BTL_DATE,BTL_TIME
TEST,1,1,1,0,0,20220101,0000,0,0,2,0,20220101,0000"""
    )
    ds = read_csv(test_data)

    assert "pressure" in ds
