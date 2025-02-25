import pytest

from cchdo.hydro.checks import check_ancillary_variables


def test_check_ancillary_variables_ok(nc_placeholder):
    # passes if it doesn't raise
    check_ancillary_variables(nc_placeholder)


def test_check_ancillary_variables_unreferenced_vars(nc_placeholder):
    with pytest.raises(ValueError):
        data = nc_placeholder.copy()
        del data.del_carbon_14_dic.attrs["ancillary_variables"]
        check_ancillary_variables(data)


def test_check_ancillary_variables_referenced_non_extant_var(nc_placeholder):
    with pytest.raises(ValueError):
        data = nc_placeholder.copy()
        del data["del_carbon_14_dic_qc"]
        check_ancillary_variables(data)
