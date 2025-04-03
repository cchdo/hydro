from enum import StrEnum, auto
from typing import Literal

FileTypes = Literal["C", "B"]


class FileType(StrEnum):
    CTD = "C"
    BOTTLE = "B"


FileTypeType = FileTypes | FileType

PrecisionSources = Literal["file", "database"]


class PrecisionSource(StrEnum):
    FILE = auto()
    DATABASE = auto()


PrecisionSourceType = PrecisionSources | PrecisionSource

WHPNameAttr = str | list[str]
