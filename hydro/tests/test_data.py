import pytest

CF_VERSION = "67"


def test_cf_standard_names():
    from hydro import data

    assert "cf_standard_name_table_version" in data.__versions__
    assert "cf_standard_name_table_date" in data.__versions__


def test_cf_standard_name_version():
    from hydro import data

    assert data.__versions__["cf_standard_name_table_version"] == CF_VERSION


cf_name_data = [
    ("sea_water_practical_salinity", "1"),
    ("sea_water_pressure", "dbar"),
    ("moles_of_oxygen_per_unit_mass_in_sea_water", "mol kg-1"),
]


@pytest.mark.parametrize("name,unit", cf_name_data)
def test_a_few_cf_standard_names(name, unit):
    from hydro import data

    assert name in data.cf_standard_names
    assert isinstance(data.cf_standard_names[name], data.CFStandardName)
    assert data.cf_standard_names[name].canonical_units == unit


cf_alias_data = [
    ("sea_floor_depth", "sea_floor_depth_below_geoid"),
    (
        "moles_per_unit_mass_of_cfc11_in_sea_water",
        "moles_of_cfc11_per_unit_mass_in_sea_water",
    ),
]


@pytest.mark.parametrize("alias,canonical", cf_alias_data)
def test_cf_standard_name_alias(alias, canonical):
    from hydro import data

    assert alias in data.cf_standard_names
    assert canonical in data.cf_standard_names

    assert data.cf_standard_names[alias] == data.cf_standard_names[canonical]
