import pytest
from hydro import data

CF_VERSION = "70"


def _argo_cf_names():
    known_bad = {
        "upwelling_radiance_in_sea_water",
    }
    for key, name in data.ArgoNames.items():
        if name.cf_standard_name in known_bad:
            reason = f"{name.cf_standard_name} not in standard name list"
            marks = pytest.mark.xfail(reason=reason)
            param = pytest.param(name, marks=marks, id=key)

            yield param

        elif name.cf_standard_name is not None:
            yield pytest.param(name, id=key)


def _argo_whp_names():
    for key, name in data.ArgoNames.items():
        if name.whp is not None:
            yield pytest.param(name, id=key)


def _whp_argo_names():
    for key, name in data.WHPNames.items():
        if name.argo is not None:
            yield pytest.param(name, id=f"{key[0]} [{key[1]}]")


def test_lengths():
    assert len(data.CFStandardNames) > 1
    assert len(data.WHPNames) > 1
    assert len(data.ArgoNames) > 1


def test_cf_names_self():
    for name in data.CFStandardNames.values():
        assert name.cf is name


def test_cf_standard_names():
    data.CFStandardNames._load_data()
    assert "cf_standard_name_table_version" in data.CFStandardNames.__versions__
    assert "cf_standard_name_table_date" in data.CFStandardNames.__versions__


def test_cf_standard_name_version():
    data.CFStandardNames._load_data()
    assert (
        data.CFStandardNames.__versions__["cf_standard_name_table_version"]
        == CF_VERSION
    )


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


argo_cf_names = list(_argo_cf_names())


@pytest.mark.parametrize("argoname", argo_cf_names)
def test_argo_cf_names_in_cf_list(argoname):
    assert argoname.cf_standard_name in data.CFStandardNames


@pytest.mark.parametrize("argoname", argo_cf_names)
def test_argo_cf_property(argoname):
    """For the list of argonames that have cf names, lets exercise the machinery
    which returns the actual CFStandardName instance"""
    assert isinstance(argoname.cf, data.CFStandardName)


argo_whp_names = list(_argo_whp_names())


@pytest.mark.parametrize("argoname", argo_whp_names)
def test_argo_whp(argoname):
    assert len(argoname.whp) >= 1
    for whp in argoname.whp:
        assert isinstance(whp, data.WHPName)


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


whp_argo_names = _whp_argo_names()


@pytest.mark.parametrize("whpname", whp_argo_names)
def test_whp_argo(whpname):
    assert len(whpname.argo) >= 1
    for argo in whpname.argo:
        assert isinstance(argo, data.ArgoName)
