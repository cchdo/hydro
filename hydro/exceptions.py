class ExchangeError(ValueError):
    pass


class ExchangeBottleError(ExchangeError):
    pass


class ExchangeCTDError(ExchangeError):
    pass


class ExchangeEncodingError(ExchangeError):
    pass


class ExchangeBOMError(ExchangeError):
    pass


class ExchangeLEError(ExchangeError):
    pass


class ExchangeMagicNumberError(ExchangeError):
    pass


class ExchangeEndDataError(ExchangeError):
    pass


class ExchangeParameterError(ExchangeError):
    pass


class ExchangeParameterUndefError(ExchangeParameterError):
    pass


class ExchangeParameterUnitAlignmentError(ExchangeParameterError):
    pass


class ExchangeDuplicateParameterError(ExchangeParameterError):
    pass


class ExchangeOrphanFlagError(ExchangeParameterError):
    pass


class ExchangeFlaglessParameterError(ExchangeParameterError):
    pass


class ExchangeFlagUnitError(ExchangeParameterError):
    pass


class ExchangeDataError(ExchangeError):
    pass


class ExchangeDataColumnAlignmentError(ExchangeDataError):
    pass


class ExchangeDataFlagPairError(ExchangeDataError):
    """There is a mismatch between what the flag value expects, and the fill/data value"""


class ExchangeDataPartialKeyError(ExchangeDataError):
    """A part of the composite key is missing"""


class ExchangeDuplicateKeyError(ExchangeDataError):
    """There is a duplicate composite key in the file"""


class ExchangeDataPartialCoordinateError(ExchangeDataError):
    """There is not enough information determine space and time coordinates"""


class ExchangeDataInconsistentCoordinateError(ExchangeDataError):
    """More than one lat, lon, or time reported for a single profile"""
