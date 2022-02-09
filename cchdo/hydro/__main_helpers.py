import os

from .exchange.two_pass import read_exchange
from cchdo.hydro.accessors import register

register()


def p_file(file_m):
    t_dir, file, file_metadata = file_m
    try:
        ex_xr = read_exchange(file)  # , parallelize=False).to_xarray()
        to_path = os.path.join(t_dir, ex_xr.cchdo.gen_fname())
        ex_xr.to_netcdf(to_path)
        return (200, to_path, file_metadata)
    except (ValueError, KeyError) as err:
        return (500, repr(err), file_metadata)
