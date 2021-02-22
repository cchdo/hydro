import xarray as xr

from hydro.rename import rename_with_bookkeeping, to_argo_variable_names


def is_not_none(a):
    return a is not None


def test_rename():
    example = xr.Dataset(
        {
            "var0": xr.DataArray(
                [],
                attrs={
                    "standard_name": "cf_stdname1",
                    "ancillary_variables": "var1 var2",
                },
            ),
            "var1": xr.DataArray([], attrs={"standard_name": "cf_stdname2"}),
            "var2": xr.DataArray([], attrs={"standard_name": "cf_stdname3"}),
        }
    )
    name_dict = {"var1": "new_var1"}
    result = rename_with_bookkeeping(example, name_dict, attrs=["ancillary_variables"])
    assert "new_var1" in result.data_vars.keys()
    for var, da in result.filter_by_attrs(ancillary_variables=is_not_none).items():
        assert "new_var1" in da.attrs["ancillary_variables"]
        assert "var2" in da.attrs["ancillary_variables"]


def test_to_argo_variable_names():
    to_argo_variable_names("TODO")
