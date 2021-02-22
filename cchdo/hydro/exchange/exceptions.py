class ExchangeError(ValueError):
    """This is the base exception which all the other exceptions derive from.
    It is a subclass of ValueError.
    """


# class ExchangeBottleError(ExchangeError):
#     pass
#
#
# class ExchangeCTDError(ExchangeError):
#     pass


class ExchangeEncodingError(ExchangeError):
    """Error raised when the bytes for some exchange file cannot be decoded as UTF-8."""


class ExchangeBOMError(ExchangeError):
    """Error raised when the exchange file has a byte order mark."""


class ExchangeLEError(ExchangeError):
    """Error raised when the exchange file does not have the correct line endings."""


class ExchangeMagicNumberError(ExchangeError):
    """Error raised when the exchange file does not start with ``BOTTLE`` or ``CTD``."""


class ExchangeEndDataError(ExchangeError):
    """Erorr raised when ``END_DATA`` cannot be found in the exchange file."""


class ExchangeParameterError(ExchangeError):
    """Base exception for errors related to parameters and units."""


class ExchangeParameterUndefError(ExchangeParameterError):
    """Error raised when the library does not have a definition for a parameter/unit pair in the exchange file."""


class ExchangeParameterUnitAlignmentError(ExchangeParameterError):
    """Error raised when there is a mismatch between the number of parameters and number of units in the exchange file."""


class ExchangeDuplicateParameterError(ExchangeParameterError):
    """Error raised when the same parameter/unit pair occurs more than once in the excahnge file."""


class ExchangeOrphanFlagError(ExchangeParameterError):
    """Error raised when there exists a flag column with no corresponding parameter column."""


class ExchangeFlaglessParameterError(ExchangeParameterError):
    """Error raised when a parameter has a flag column when it is not supposed to."""


class ExchangeFlagUnitError(ExchangeParameterError):
    """Error raised if a flag column has a non empty units."""


class ExchangeDataError(ExchangeError):
    """Base exception for errors which occur when parsing the data porition of an exchange file."""


class ExchangeDataColumnAlignmentError(ExchangeDataError):
    """Error raised when the number of columns in a data line does not match the expected number of columns based on the parameter/unit lines."""


class ExchangeDataFlagPairError(ExchangeDataError):
    """There is a mismatch between what the flag value expects, and the fill/data value.

    Examples:

    * something with a flag of ``9`` has a non fill value
    * something with a flag of ``2`` as a fill value instead of data
    """


class ExchangeDataPartialKeyError(ExchangeDataError):
    """Error raised when there is no value for one (or more) of the following parameters.

    * EXPOCODE
    * STNNBR
    * CASTNO
    * SAMPNO (only for bottle files)
    * CTDPRS (only for CTD files)

    These form the "composite key" which uniquely identify the "row" of exchange data.
    """


class ExchangeDuplicateKeyError(ExchangeDataError):
    """Error raised when there is a duplicate composite key in the exchange file.

    This would occur if the exact values for the following parameters occur in more than one data row:

    * EXPOCODE
    * STNNBR
    * CASTNO
    * SAMPNO (only for bottle files)
    * CTDPRS (only for CTD files)
    """


class ExchangeDataPartialCoordinateError(ExchangeDataError):
    """Error raised if values for latitude, longitude, or date are missing.

    It is OK by the standard to omit the time of day.
    """


class ExchangeDataInconsistentCoordinateError(ExchangeDataError):
    """Error raised if the reported latitude, longitude, and date (and time) vary for a single profile.

    A "profile" in an exchange file is a grouping of data rows which all have the same EXPOCODE, STNNBR, and CASTNO.
    The SAMPNO/CTDPRS is allowed/requried to vary for a single profile and is what identifies samples within one profile.
    """


class ExchangeInconsistentMergeType(ExchangeError):
    """Error raised when the merge_ex method is called on mixed ctd and bottle exchange types"""


class ExchangeRecursiveZip(ExchangeError):
    """Error raised if there are zip files inside the zip file that read exchange is trying to read"""
