import os
import warnings

import xarray as xr

from cchdo.hydro import accessors  # noqa
from cchdo.hydro.exchange import read_exchange
from cchdo.hydro.exchange.exceptions import (
    ExchangeDataFlagPairError,
    ExchangeParameterUndefError,
)
from cchdo.params import WHPNames


def p_file(file_m):
    t_dir, file, file_metadata = file_m
    checks = {"flags": False}
    unknown_params = []

    warnings.simplefilter("ignore")
    if ("OSNUM", None) not in WHPNames:
        WHPNames.add_alias(("OSNUM", None), ("EVENT_NUMBER", None))

    if file_metadata["data_type"] == "ctd" and ("NITRATE", "UMOL/KG") not in WHPNames:
        # HOT names that are a little dangerous to have in the real DB
        WHPNames.add_alias(("NITRATE", "UMOL/KG"), ("CTDNITRATE", "UMOL/KG"))
        WHPNames.add_alias(("CHLPIG", "uG/L"), ("CTDFLUOR", "MG/M^3"))
        WHPNames.add_alias(("CHLPIG", "UG/L"), ("CTDFLUOR", "MG/M^3"))

        # other one off things
        WHPNames.add_alias(("CTDFLUOR", "UG/L_UNCALIBRATED"), ("CTDFLUOR", "MG/M^3"))
        WHPNames.add_alias(("_OS_ID", None), ("EVENT_NUMBER", None))
        WHPNames.add_alias(("CTDSAL", "PPS-78"), ("CTDSAL", "PSS-78"))

    try:
        ex_xr = read_exchange(file, checks=checks)
    except ExchangeParameterUndefError as err:
        return (500, repr(err), file_metadata, err.error_data)
    except ExchangeDataFlagPairError:
        try:
            ex_xr = read_exchange(file, fill_values=("-999", "-99"), checks=checks)
        except (ValueError, KeyError) as err:
            return (500, repr(err), file_metadata, unknown_params)
    except (ValueError, KeyError) as err:
        return (500, repr(err), file_metadata, unknown_params)

    to_path = os.path.join(t_dir, f"{file_metadata['id']}_{ex_xr.cchdo.gen_fname()}")
    ex_xr.to_netcdf(to_path)
    return (200, to_path, file_metadata, unknown_params)


def p_file_cf(file_m):
    t_dir, file, file_metadata = file_m

    warnings.simplefilter("ignore")

    ex_xr = xr.load_dataset(file)
    try:
        ex_xr.cchdo.to_exchange()
        exchange_ok = True
    except Exception:
        exchange_ok = False

    try:
        ex_xr.cchdo.to_coards()
        coards_ok = True
    except Exception:
        coards_ok = False

    try:
        ex_xr.cchdo.to_woce()
        woce_ok = True
    except Exception:
        woce_ok = False

    try:
        ex_xr.cchdo.to_sum()
        sum_ok = True
    except Exception:
        sum_ok = False

    return (file_metadata, exchange_ok, coards_ok, woce_ok, sum_ok)
