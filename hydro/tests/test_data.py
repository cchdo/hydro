import pytest
import cchdo.params as data

CF_VERSION = "72"


def test_lengths():
    assert len(data.CFStandardNames) > 1
    assert len(data.WHPNames) > 1


def test_cf_names_self():
    for name in data.CFStandardNames.values():
        assert name.cf is name


cf_name_data = [
    ("sea_water_practical_salinity", "1"),
    ("sea_water_pressure", "dbar"),
    ("moles_of_oxygen_per_unit_mass_in_sea_water", "mol kg-1"),
]


@pytest.mark.parametrize("name,unit", cf_name_data)
def test_a_few_cf_standard_names(name, unit):
    assert name in data.CFStandardNames
    assert isinstance(data.CFStandardNames[name], data.CFStandardName)
    assert data.CFStandardNames[name].canonical_units == unit


cf_alias_data = [
    ("sea_floor_depth", "sea_floor_depth_below_geoid"),
    (
        "moles_per_unit_mass_of_cfc11_in_sea_water",
        "moles_of_cfc11_per_unit_mass_in_sea_water",
    ),
]


# @pytest.mark.parametrize("alias,canonical", cf_alias_data)
# def test_cf_standard_name_alias(alias, canonical):
#    assert alias in data.CFStandardNames
#    assert canonical in data.CFStandardNames
#
#    assert data.CFStandardNames[alias] == data.CFStandardNames[canonical]


whp_cf_names = [value for value in data.WHPNames.values() if value.cf_name is not None]


@pytest.mark.parametrize("whpname", whp_cf_names)
def test_whp_cf_names_in_cf_list(whpname):
    assert whpname.cf_name in data.CFStandardNames


whp_error_names = [
    name for name in data.WHPNames.values() if name.error_name is not None
]


@pytest.mark.parametrize("whpname", whp_error_names)
def test_whp_error_names(whpname):
    assert data.WHPNames.error_cols[whpname.error_name] is whpname


whp_unitless_names = [name for name in data.WHPNames.values() if name.whp_unit is None]


@pytest.mark.parametrize("whpname", whp_unitless_names)
def test_whp_no_unit_params(whpname):
    str_name = whpname.whp_name
    tuple_name = (str_name,)

    assert data.WHPNames[str_name] == whpname
    assert data.WHPNames[tuple_name] == whpname


@pytest.mark.parametrize("whpname", whp_cf_names)
def test_whp_cf_property(whpname):
    assert isinstance(whpname.cf, data.CFStandardName)
