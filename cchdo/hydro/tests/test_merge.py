from .. import accessors  # noqa


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
