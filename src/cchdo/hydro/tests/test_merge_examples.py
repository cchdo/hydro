import json
from importlib.resources import path, read_text

import pytest
import xarray as xr

from cchdo.hydro import accessors  # noqa: F401

real_fq_samples = pytest.importorskip(
    "cchdo.hydro.tests.data.real_fq_examples",
    reason="Real FQ merge examples could not be found, this is expected with an installation from pypi",
)


def remove_unstable_attrs(ds: xr.Dataset) -> xr.Dataset:
    unstable_attrs = (
        "date_modified",
        "data_metadata_modified",
    )
    out = ds.copy()
    for attr in unstable_attrs:
        for var in out.variables:
            try:
                del out[var].attrs[attr]
            except KeyError:
                continue
        try:
            del ds.attrs[attr]
        except KeyError:
            continue
    return out


@pytest.mark.parametrize(
    "input_dir", ["fq_33RO20200321", "fq_320620170703", "fq_325020240221"]
)
def test_real_fq(input_dir):
    fq = json.loads(read_text(real_fq_samples, f"{input_dir}/fq.json"))
    with path(real_fq_samples, f"{input_dir}/in.nc") as fi:
        in_nc = remove_unstable_attrs(xr.load_dataset(fi))
    with path(real_fq_samples, f"{input_dir}/out.nc") as fi:
        out_nc = remove_unstable_attrs(xr.load_dataset(fi))

    merged = remove_unstable_attrs(
        in_nc.cchdo.merge_fq(fq, check_flags=False)
    )  # we don't care about flag validity for this test
    xr.testing.assert_identical(merged, out_nc)
