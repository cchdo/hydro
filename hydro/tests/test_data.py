import pytest
from hydro import data

CF_VERSION = "67"


def test_cf_standard_names():
    assert "cf_standard_name_table_version" in data.__versions__
    assert "cf_standard_name_table_date" in data.__versions__


def test_cf_standard_name_version():
    assert data.__versions__["cf_standard_name_table_version"] == CF_VERSION


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


@pytest.mark.parametrize("alias,canonical", cf_alias_data)
def test_cf_standard_name_alias(alias, canonical):
    assert alias in data.CFStandardNames
    assert canonical in data.CFStandardNames

    assert data.CFStandardNames[alias] == data.CFStandardNames[canonical]


argo_cf_names = [
    value for value in data.ArgoNames.values() if value.cf_standard_name is not None
]


@pytest.mark.parametrize("argoname", argo_cf_names)
def test_argo_cf_names_in_cf_list(argoname):
    if argoname.cf_standard_name == "upwelling_radiance_in_sea_water":
        pytest.xfail(
            "upwelling_radiance_in_sea_water is not in the standard names list"
        )
    assert argoname.cf_standard_name in data.CFStandardNames


whp_cf_names = [value for value in data.WHPNames.values() if value.cf_name is not None]


@pytest.mark.parametrize("whpname", whp_cf_names)
def test_whp_cf_names_in_cf_list(whpname):
    assert whpname.cf_name in data.CFStandardNames
