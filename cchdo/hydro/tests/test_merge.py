import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cchdo.hydro import accessors  # noqa
from cchdo.hydro.exchange.exceptions import ExchangeDataFlagPairError


def test_fq_merge(nc_placeholder):
    fq = [
        {
            "EXPOCODE": "TEST",
            "STNNBR": "1",
            "CASTNO": 1,
            "SAMPNO": "1",
            "OXYGEN [UMOL/KG]": "234.1",
            "OXYGEN [UMOL/KG]_FLAG_W": "2",
        }
    ]
    merged = nc_placeholder.cchdo.merge_fq(fq)

    assert merged.oxygen.data == [[234.1]]
    assert merged.oxygen_qc.data == [[2]]
    assert merged.oxygen.attrs["C_format"] == "%.1f"
    assert merged.oxygen.attrs["C_format_source"] == "input_file"


def test_fq_merge_with_error(nc_placeholder):
    fq = [
        {
            "EXPOCODE": "TEST",
            "STNNBR": "1",
            "CASTNO": 1,
            "SAMPNO": "1",
            "DELC14 [/MILLE]": "-3.12",
            "DELC14 [/MILLE]_FLAG_W": "2",
            "C14ERR [/MILLE]": "0.12",
        }
    ]
    merged = nc_placeholder.cchdo.merge_fq(fq)

    assert merged.del_carbon_14_dic.data == [[-3.12]]
    assert merged.del_carbon_14_dic_qc.data == [[2]]
    assert merged.del_carbon_14_dic_error.data == [[0.12]]
    assert merged.del_carbon_14_dic_error.attrs["C_format"] == "%.2f"
    assert merged.del_carbon_14_dic_error.attrs["C_format_source"] == "input_file"


def test_fq_merge_flag_error(nc_placeholders):
    fq = [
        {
            "EXPOCODE": "TEST",
            "STNNBR": "1",
            "CASTNO": 1,
            "SAMPNO": "36",
            "OXYGEN [UMOL/KG]": "200.1",
        }
    ]
    with pytest.raises(ExchangeDataFlagPairError):
        nc_placeholders.cchdo.merge_fq(fq)


def test_fq_merge_flag_only(nc_placeholders):
    # Ensure that a valid flag only update succeeds
    # 9 -> 1 for missing data
    # 1 -> 9 for missing data
    # 1 -> 5 for missing data
    # 2 -> 3 in the case of extant data

    # 9 -> 1
    fq = [
        {
            "EXPOCODE": "TEST",
            "STNNBR": "1",
            "CASTNO": 1,
            "SAMPNO": "36",
            "OXYGEN [UMOL/KG]_FLAG_W": "1",
        }
    ]
    merged = nc_placeholders.cchdo.merge_fq(fq)
    assert_array_equal(merged.oxygen_qc.data, [[1, 1, 5, np.nan, 1, 2]])

    # 1 -> 9
    fq = [
        {
            "EXPOCODE": "TEST",
            "STNNBR": "1",
            "CASTNO": 1,
            "SAMPNO": "35",
            "OXYGEN [UMOL/KG]_FLAG_W": None,
        }
    ]
    merged = nc_placeholders.cchdo.merge_fq(fq)
    assert_array_equal(merged.oxygen_qc.data, [[np.nan, np.nan, 5, np.nan, 1, 2]])

    # 1 -> 5
    fq = [
        {
            "EXPOCODE": "TEST",
            "STNNBR": "1",
            "CASTNO": 1,
            "SAMPNO": "35",
            "OXYGEN [UMOL/KG]_FLAG_W": "5",
        }
    ]
    merged = nc_placeholders.cchdo.merge_fq(fq)
    assert_array_equal(merged.oxygen_qc.data, [[np.nan, 5, 5, np.nan, 1, 2]])

    # 2 -> 3
    fq = [
        {
            "EXPOCODE": "TEST",
            "STNNBR": "1",
            "CASTNO": 1,
            "SAMPNO": "30",
            "OXYGEN [UMOL/KG]_FLAG_W": "3",
        }
    ]
    merged = nc_placeholders.cchdo.merge_fq(fq)
    assert_array_equal(merged.oxygen_qc.data, [[np.nan, 1, 5, np.nan, 1, 3]])
