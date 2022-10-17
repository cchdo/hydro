FILL_VALUE = -9


ASTERISK_FLAG = "*" * 7


CHARACTER_PARAMETERS = ["STNNBR", "SAMPNO", "BTLNBR"]


COLUMN_WIDTH = 8
SAFE_COLUMN_WIDTH = COLUMN_WIDTH - 1


UNKNONW_TIME_FILL = "0000"


BOTTLE_FLAGS = {
    1: "Bottle information unavailable.",
    2: "No problems noted.",
    3: "Leaking.",
    4: "Did not trip correctly.",
    5: "Not reported.",
    6: (
        "Significant discrepancy in measured values between Gerard and Niskin "
        "bottles."
    ),
    7: "Unknown problem.",
    8: (
        "Pair did not trip correctly. Note that the Niskin bottle can trip at "
        "an unplanned depth while the Gerard trips correctly and vice versa."
    ),
    9: "Samples not drawn from this bottle.",
}


CTD_FLAGS = {
    1: "Not calibrated",
    2: "Acceptable measurement",
    3: "Questionable measurement",
    4: "Bad measurement",
    5: "Not reported",
    6: "Interpolated over >2 dbar interval",
    7: "Despiked",
    8: "Not assigned for CTD data",
    9: "Not sampled",
}


WATER_SAMPLE_FLAGS = {
    1: (
        "Sample for this measurement was drawn from water bottle but analysis "
        "not received."
    ),
    2: "Acceptable measurement.",
    3: "Questionable measurement.",
    4: "Bad measurement.",
    5: "Not reported.",
    6: "Mean of replicate measurements.",
    7: "Manual chromatographic peak measurement.",
    8: "Irregular digital chromatographic peak integration.",
    9: "Sample not drawn for this measurement from this bottle.",
}


def flag_description(flag_map):
    return ":".join(
        [":"]
        + ["%d = %s" % (i + 1, flag_map[i + 1]) for i in range(len(flag_map))]
        + ["\n"]
    )


BOTTLE_FLAG_DESCRIPTION = flag_description(BOTTLE_FLAGS)


CTD_FLAG_DESCRIPTION = flag_description(CTD_FLAGS)


WATER_SAMPLE_FLAG_DESCRIPTION = ":".join(
    [":"]
    + [
        "%d = %s" % (i + 1, WATER_SAMPLE_FLAGS[i + 1])
        for i in range(len(WATER_SAMPLE_FLAGS))
    ]
    + ["\n"]
)
