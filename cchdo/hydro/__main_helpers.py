import os
import warnings

from .exchange import read_exchange
from cchdo.params import WHPNames
from .exchange.exceptions import ExchangeDataFlagPairError, ExchangeParameterUndefError
from . import accessors  # noqa


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
