import os

from .exchange.two_pass import read_exchange
from cchdo.hydro.accessors import register
from cchdo.params import WHPNames

register()


def p_file(file_m):
    t_dir, file, file_metadata = file_m
    fill_values = ("-999",)
    if file_metadata["data_type"] == "ctd" and ("NITRATE", "UMOL/KG") not in WHPNames:

        # HOT names that are a little dangerous to have in the real DB
        WHPNames.add_alias(("NITRATE", "UMOL/KG"), ("CTDNITRATE", "UMOL/KG"))
        WHPNames.add_alias(("CHLPIG", "uG/L"), ("CTDFLUOR", "MG/M^3"))
        WHPNames.add_alias(("CHLPIG", "UG/L"), ("CTDFLUOR", "MG/M^3"))

        # other one off things
        WHPNames.add_alias(("CTDFLUOR", "UG/L_UNCALIBRATED"), ("CTDFLUOR", "MG/M^3"))
        WHPNames.add_alias(("_OS_ID", None), ("EVENT_NUMBER", None))
        WHPNames.add_alias(("CTDSAL", "PPS-78"), ("CTDSAL", "PSS-78"))

    if file_metadata["data_type"] == "ctd":
        fill_values = ("-999", "-99")

    try:
        ex_xr = read_exchange(
            file, fill_values=fill_values
        )  # , parallelize=False).to_xarray()
        to_path = os.path.join(
            t_dir, f"{file_metadata['id']}_{ex_xr.cchdo.gen_fname()}"
        )
        ex_xr.to_netcdf(to_path)
        return (200, to_path, file_metadata)
    except (ValueError, KeyError) as err:
        return (500, repr(err), file_metadata)
