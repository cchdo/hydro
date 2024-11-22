import cchdo.hydro.legacy.woce as legacy_woce


def test_woce_flag_defs():
    """This test was made to ensure that a refactor produced the same result"""

    BOTTLE_FLAG_DESCRIPTION = "::1 = Bottle information unavailable.:2 = No problems noted.:3 = Leaking.:4 = Did not trip correctly.:5 = Not reported.:6 = Significant discrepancy in measured values between Gerard and Niskin bottles.:7 = Unknown problem.:8 = Pair did not trip correctly. Note that the Niskin bottle can trip at an unplanned depth while the Gerard trips correctly and vice versa.:9 = Samples not drawn from this bottle.:\n"
    CTD_FLAG_DESCRIPTION = "::1 = Not calibrated:2 = Acceptable measurement:3 = Questionable measurement:4 = Bad measurement:5 = Not reported:6 = Interpolated over >2 dbar interval:7 = Despiked:8 = Not assigned for CTD data:9 = Not sampled:\n"
    WATER_SAMPLE_FLAG_DESCRIPTION = "::1 = Sample for this measurement was drawn from water bottle but analysis not received.:2 = Acceptable measurement.:3 = Questionable measurement.:4 = Bad measurement.:5 = Not reported.:6 = Mean of replicate measurements.:7 = Manual chromatographic peak measurement.:8 = Irregular digital chromatographic peak integration.:9 = Sample not drawn for this measurement from this bottle.:\n"

    assert legacy_woce.BOTTLE_FLAG_DESCRIPTION == BOTTLE_FLAG_DESCRIPTION
    assert legacy_woce.CTD_FLAG_DESCRIPTION == CTD_FLAG_DESCRIPTION
    assert legacy_woce.WATER_SAMPLE_FLAG_DESCRIPTION == WATER_SAMPLE_FLAG_DESCRIPTION
