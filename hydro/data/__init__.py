from dataclasses import dataclass, field
from xml.etree import ElementTree
from importlib.resources import read_text, open_text
from typing import Optional
from types import MappingProxyType
from csv import DictReader

__all__ = ["CFStandardNames"]

__versions__ = {}  # pile of data version infomation


@dataclass(frozen=True)
class CFStandardName:
    """Wrapper for CF Standard Names"""

    name: str  # is the 'id' property in the xml
    canonical_units: str
    grib: Optional[str]
    amip: Optional[str]
    description: str = field(repr=False, hash=False)


@dataclass(frozen=True)
class ArgoName:
    """Wrapper for Argo variable name table
    Note that most of the table is ignored, this 
    is here to mostly map CF names to argo and back
    """

    name: str
    order: int
    cf_standard_name: Optional[str]
    unit: str
    fillvalue: str
    dtype: str


def _load_cf_standard_names():
    cf_standard_names = {}

    for element in ElementTree.fromstring(
        read_text("hydro.data", "cf-standard-name-table.xml")
    ):
        if element.tag == "version_number":
            __versions__["cf_standard_name_table_version"] = element.text

        if element.tag == "last_modified":
            __versions__["cf_standard_name_table_date"] = element.text

        if element.tag not in ("entry", "alias"):
            continue

        name = element.attrib["id"]
        name_info = {info.tag: info.text for info in element}

        if element.tag == "entry":
            cf_standard_names[name] = CFStandardName(name=name, **name_info)

        if element.tag == "alias":
            cf_standard_names[name] = cf_standard_names[name_info["entry_id"]]

    return cf_standard_names


def _load_argo_names():
    argo_names = {}
    with open_text(
        "hydro.data", "argo-parameters-list-code-and-b.csv", encoding="utf-8-sig"
    ) as f:
        for row in DictReader(f):
            if row["cf_standard_name"] == "-":
                row["cf_standard_name"] = None
            argo_names[row["parameter name"]] = ArgoName(
                name=row["parameter name"],
                order=int(row["Order"]),
                cf_standard_name=row["cf_standard_name"],
                unit=row["unit"],
                fillvalue=row["Fillvalue"],
                dtype=row["Data Type"],
            )
    return argo_names


def _load_whp_names():
    ...


CFStandardNames = MappingProxyType(_load_cf_standard_names())
ArgoNames = MappingProxyType(_load_argo_names())
WHPNames = ...
