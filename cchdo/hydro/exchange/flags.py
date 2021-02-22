"""A Collection of Flag Schemes"""
from enum import IntEnum


class ExchangeBottleFlag(IntEnum):
    """Enum representing a WHP Bottle flag

    This flag represents information about the sampling device itself
    (i.e. the niskin bottle). It should only be used for "BTLNBR_FLAG_W"
    values and should never be used with CTD files.
    """

    NOFLAG = 0  # no idea if this will cause issue
    NO_INFO = 1
    GOOD = 2
    LEAKING = 3
    BAD_TRIP = 4
    NOT_REPORTED = 5
    DISCREPANCY = 6
    UNKNOWN = 7
    PAIR = 8
    NOT_SAMPLED = 9

    def __init__(self, flag):
        self.flag = flag

    @property
    def definition(self):
        """Prints the human readable flag definition"""

        defs = {
            0: "No Flag assigned",
            1: "Bottle information unavailable.",
            2: "No problems noted.",
            3: "Leaking.",
            4: "Did not trip correctly.",
            5: "Not reported.",
            6: "Significant discrepancy in measured values between Gerard and Niskin bottles.",  # noqa: E501
            7: "Unknown problem.",
            8: "Pair did not trip correctly. Note that the Niskin bottle can trip at an unplanned depth while the Gerard trips correctly and vice versa.",  # noqa: E501
            9: "Samples not drawn from this bottle.",
        }
        return defs[self.flag]

    @property
    def cf_def(self):
        return "_".join(self.definition.lower().replace(".", "").split())

    @property
    def has_value(self):
        """Should the data this is a flag for contain a value"""
        if self.flag in (1, 5, 9):
            return False
        return True


class ExchangeSampleFlag(IntEnum):
    NOFLAG = 0  # no idea if this will cause issue
    MISSING = 1
    GOOD = 2
    QUESTIONABLE = 3
    BAD = 4
    NOT_REPORTED = 5
    MEAN = 6
    CHROMA_MANUAL = 7
    CHROMA_IRREGULAR = 8
    NOT_SAMPLED = 9

    __doc__ = """Test Doc String"""

    def __init__(self, flag):
        self.flag = flag

    @property
    def definition(self):
        defs = {
            0: "No Flag assigned",
            1: "Sample for this measurement was drawn from water bottle but analysis not received.",  # noqa: E501
            2: "Acceptable measurement.",
            3: "Questionable measurement.",
            4: "Bad measurement.",
            5: "Not reported.",
            6: "Mean of replicate measurements",
            7: "Manual chromatographic peak measurement.",
            8: "Irregular digital chromatographic peak integration.",
            9: "Sample not drawn for this measurement from this bottle.",
        }
        return defs[self.flag]

    @property
    def cf_def(self):
        return "_".join(self.definition.lower().replace(".", "").split())

    @property
    def has_value(self):
        if self.flag in (1, 5, 9):
            return False
        return True


class ExchangeCTDFlag(IntEnum):
    NOFLAG = 0  # no idea if this will cause issue
    UNCALIBRATED = 1
    GOOD = 2
    QUESTIONABLE = 3
    BAD = 4
    NOT_REPORTED = 5
    INTERPOLATED = 6
    DESPIKED = 7
    NOT_SAMPLED = 9

    def __init__(self, flag):
        self.flag = flag

    @property
    def definition(self):
        defs = {
            0: "No Flag assigned",
            1: "Not calibrated.",
            2: "Acceptable measurement.",
            3: "Questionable measurement.",
            4: "Bad measurement.",
            5: "Not reported.",
            6: "Interpolated over a pressure interval larger than 2 dbar.",
            7: "Despiked.",
            9: "Not sampled.",
        }
        return defs[self.flag]

    @property
    def cf_def(self):
        return "_".join(self.definition.lower().replace(".", "").split())

    @property
    def has_value(self):
        if self.flag in (5, 9):
            return False
        return True
