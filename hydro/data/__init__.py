from dataclasses import dataclass, field
from xml.etree import ElementTree
from importlib.resources import read_text
from typing import Optional

__versions__ = {}  # pile of data version infomation


@dataclass
class CFStandardName:
    """Wrapper for CF Standard Names"""

    name: str  # is the 'id' property in the xml
    canonical_units: str
    grib: Optional[str]
    amip: Optional[str]
    description: str = field(repr=False, hash=False)


_cf_table = ElementTree.fromstring(
    read_text("hydro.data", "cf-standard-name-table.xml")
)

cf_standard_names = {}

for element in _cf_table:
    if element.tag == "version_number":
        __versions__["cf_standard_name_table_version"] = element.text

    if element.tag == "last_modified":
        __versions__["cf_standard_name_table_date"] = element.text

    if element.tag not in ('entry', 'alias'):
        continue

    name = element.attrib["id"]
    name_info = {info.tag: info.text for info in element}

    if element.tag == "entry":
        cf_standard_names[name] = CFStandardName(name=name, **name_info)
    
    if element.tag == "alias":
        cf_standard_names[name] = cf_standard_names[name_info['entry_id']]

del _cf_table