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
