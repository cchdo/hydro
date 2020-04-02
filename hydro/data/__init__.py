from dataclasses import dataclass, field
from xml.etree import ElementTree
from importlib.resources import read_text, open_text
from typing import Optional, Callable, Union
from csv import DictReader
from json import load
from collections.abc import Mapping
from functools import cached_property

__all__ = ["CFStandardNames", "ArgoNames", "WHPNames"]


def _name_getter(cf_name, names_list):
    if cf_name is None:
        return None

    names = list(filter(lambda x: x.cf == cf_name, names_list))

    if not any(names):
        return None

    return names


class ArgoNameMixin:
    @cached_property
    def argo(self):
        return _name_getter(self.cf, ArgoNames.values())


class WHPNameMixin:
    @cached_property
    def whp(self):
        return _name_getter(self.cf, WHPNames.values())


@dataclass(frozen=True)
class CFStandardName(ArgoNameMixin, WHPNameMixin):
    """Wrapper for CF Standard Names"""

    name: str  # is the 'id' property in the xml
    canonical_units: str
    grib: Optional[str]
    amip: Optional[str]
    description: str = field(repr=False, hash=False)

    @property
    def cf(self):
        return self


@dataclass(frozen=True)
class ArgoName(WHPNameMixin):
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

    @property
    def cf(self):
        return CFStandardNames.get(self.cf_standard_name)


@dataclass(frozen=True)
class WHPName(ArgoNameMixin):
    """Wrapper for WHP parameters.json
    """

    whp_name: str
    data_type: Callable[[str], Union[str, float, int]] = field(repr=False)
    whp_unit: Optional[str] = None
    flag_w: Optional[str] = field(default=None, repr=False)
    cf_name: Optional[str] = None
    numeric_min: Optional[float] = field(default=None, repr=False)
    numeric_max: Optional[float] = field(default=None, repr=False)
    numeric_precision: Optional[int] = field(default=None, repr=False)
    field_width: Optional[int] = field(default=None, repr=False)
    description: Optional[str] = field(default=None, repr=False)
    note: Optional[str] = field(default=None, repr=False)
    warning: Optional[str] = field(default=None, repr=False)
    error_name: Optional[str] = field(default=None, repr=False)
    scope: str = field(default="sample", repr=False)

    @property
    def key(self):
        """This is the thing that uniquely identifies"""
        return (self.whp_name, self.whp_unit)

    @property
    def cf(self):
        return CFStandardNames.get(self.cf_name)


def _load_cf_standard_names(__versions__):
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
    _dtype_map = {"string": str, "decimal": float, "integer": int}
    whp_name = {}
    with open_text("hydro.data", "parameters.json") as f:
        for record in load(f):
            record["data_type"] = _dtype_map[record["data_type"]]
            param = WHPName(**record)
            whp_name[param.key] = param
    # load the aliases
    with open_text("hydro.data", "aliases.json") as f:
        for record in load(f):
            whp_name[(record["whp_name"], record["whp_unit"])] = whp_name[
                (record["canonical_name"], record["canonical_unit"])
            ]

    return whp_name


class _LazyMapping(Mapping):
    _cached_dict_internal = None

    def __init__(self, loader):
        self._loader = loader

    @property
    def _cached_dict(self):
        if not self._cached_dict_internal:
            self._cached_dict_internal = self._loader()
        return self._cached_dict_internal

    def _load_data(self):
        self._cached_dict

    def __getitem__(self, key):
        return self._cached_dict[key]

    def __iter__(self):
        for key in self._cached_dict:
            yield key

    def __len__(self):
        return len(self._cached_dict)


class _WHPNames(_LazyMapping):
    def __getitem__(self, key):
        if isinstance(key, str):
            key = (key, None)

        if isinstance(key, tuple) and len(key) == 1:
            key = (*key, None)

        return self._cached_dict[key]

    @property
    def error_cols(self):
        return {
            ex.error_name: ex
            for ex in self._cached_dict.values()
            if ex.error_name is not None
        }


class _CFStandardNames(_LazyMapping):
    def __init__(self, loader):
        self._loader = loader
        self.__versions__ = {}

    @property
    def _cached_dict(self):
        if not self._cached_dict_internal:
            self._cached_dict_internal = self._loader(self.__versions__)
        return self._cached_dict_internal


CFStandardNames = _CFStandardNames(_load_cf_standard_names)
ArgoNames = _LazyMapping(_load_argo_names)
WHPNames = _WHPNames(_load_whp_names)
