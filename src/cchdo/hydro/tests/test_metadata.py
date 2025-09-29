from io import BytesIO

import pytest

from cchdo.hydro import read_csv
from cchdo.hydro.metadata import validate


def test_invalid_createor_name():
    test_data = BytesIO(
        b"""EXPOCODE,STNNBR,CASTNO,SAMPNO,LATITUDE,LONGITUDE,DATE,TIME,CTDPRS [DBAR],SILCAT [UMOL/KG],SILCAT [UMOL/KG]_FLAG_W,PHSPHT [UMOL/KG],PHSPHT [UMOL/KG]_FLAG_W
TEST,1,1,1,0,0,20220101,0000,0,0,2,0,3"""
    )
    ds = read_csv(test_data)

    ds.silicate.attrs["project"] = "nutrients"
    ds.phosphate.attrs["project"] = "nutrients"

    ds.silicate.attrs["creator_name"] = "Susan Becker"

    with pytest.raises(ExceptionGroup) as excinfo:
        validate(ds)

        assert excinfo.group_contains(ValueError)
