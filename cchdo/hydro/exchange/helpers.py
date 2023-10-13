import io

from cchdo.params import WHPNames


def simple_bottle_exchange(
    params=None, units=None, data=None, comments: str | None = None
):
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


def gen_template(
    ftype="B",
    param_counts: dict[str, int] | None = None,
    min_count=5,
    filter_erddap=False,
):
    from . import FileType, read_csv

    ftype = FileType(ftype)

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

    params = [
        "EXPOCODE",
        "STNNBR",
        "CASTNO",
        "SAMPNO",
        "LATITUDE",
        "LONGITUDE",
        "DATE",
        "TIME",
        "CTDPRS [DBAR]",
    ]
    data = ["TEST", "1", "1", "1", "0", "0", "20200101", "0000", "0"]
    for name in set(WHPNames.values()):
        if filter_erddap and not name.in_erddap:
            continue
        if name.whp_name in exclude:
            continue
        if param_counts is not None:
            if param_counts.get(name.nc_name, 0) < min_count:
                continue
        if ftype == FileType.CTD and name.flag_w in {
            "woce_discrete",
            "woce_bottle",
            "no_flags",
        }:
            continue
        if name.whp_unit is not None:
            param_name = f"{name.whp_name} [{name.whp_unit}]"
        else:
            param_name = name.whp_name
        params.append(param_name)

        data.append("-999")
        if name.flag_w is not None and name.flag_w != "no_flags":
            params.append(f"{param_name}_FLAG_W")
            data.append("9")

        if name.error_name is not None:
            if name.whp_unit is not None:
                params.append(f"{name.error_name} [{name.whp_unit}]")
            else:
                params.append(name.error_name)
            data.append("-999")

    # special case
    params.extend(["BTL_DATE", "BTL_TIME"])
    data.extend(["20220421", "0944"])

    file = f"{','.join(params)}\n{','.join(data)}"
    data_file = io.BytesIO(file.encode("utf8"))

    return read_csv(data_file, ftype=ftype)
