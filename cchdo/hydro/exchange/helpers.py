import io

from cchdo.params import WHPNames


def simple_bottle_exchange(params=None, units=None, data=None, comments: str = None):
    stamp = "BOTTLE,test"
    min_params = [
        "EXPOCODE",
        "STNNBR",
        "CASTNO",
        "SAMPNO",
        "LATITUDE",
        "LONGITUDE",
        "DATE",
        "TIME",
        "CTDPRS",
    ]
    min_units = ["", "", "", "", "", "", "", "", "DBAR"]
    min_line = ["TEST", "1", "1", "1", "0", "0", "20200101", "0000", "0"]
    end = "END_DATA"

    if params is not None:
        min_params.extend(params)
    if units is not None:
        min_units.extend(units)
    if data is not None:
        min_line.extend(data)

    if comments is not None:
        comments = "\n".join([f"#{line}" for line in comments.splitlines()])
        simple = "\n".join(
            [
                stamp,
                comments,
                ",".join(min_params),
                ",".join(min_units),
                ",".join(min_line),
                end,
            ]
        )
    else:
        simple = "\n".join(
            [stamp, ",".join(min_params), ",".join(min_units), ",".join(min_line), end]
        )
    return simple.encode("utf8")


def gen_complete_bottle(ctd_params_only=False):
    from . import read_exchange

    exclude = set(
        [
            "EXPOCODE",
            "STNNBR",
            "CASTNO",
            "SAMPNO",
            "LATITUDE",
            "LONGITUDE",
            "DATE",
            "TIME",
            "CTDPRS",
            "BTL_TIME",
            "BTL_DATE",
        ]
    )

    params = []
    units = []
    data = []
    for name in set(WHPNames.values()):
        if name.whp_name in exclude:
            continue
        if ctd_params_only is True and name.flag_w in {
            "woce_discrete",
            "woce_bottle",
            "no_flags",
        }:
            continue
        params.append(name.whp_name)
        units.append(name.whp_unit if name.whp_unit is not None else "")
        data.append("-999")
        if name.flag_w is not None and name.flag_w != "no_flags":
            params.append(f"{name.whp_name}_FLAG_W")
            units.append("")
            data.append("9")

        if name.error_name is not None:
            params.append(name.error_name)
            units.append(name.whp_unit if name.whp_unit is not None else "")
            data.append("-999")

    # special case
    params.extend(["BTL_DATE", "BTL_TIME"])
    units.extend(["", ""])
    data.extend(["20220421", "0944"])

    ex = simple_bottle_exchange(params=params, units=units, data=data)
    return read_exchange(io.BytesIO(ex))
