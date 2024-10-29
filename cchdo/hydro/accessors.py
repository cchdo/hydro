import os
import re
import string
from collections import defaultdict
from datetime import datetime, timezone
from io import BufferedWriter, BytesIO
from typing import Literal, NamedTuple
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import xarray as xr

from cchdo.params import WHPName, WHPNames

from .exchange import (
    FileType,
    add_cdom_coordinate,
    all_same,
    extract_numeric_precisions,
    flatten_cdom_coordinate,
)
from .exchange import check_flags as _check_flags

FLAG_NAME = "cchdo.hydro._qc"
ERROR_NAME = "cchdo.hydro._error"

PathType = str | bytes | os.PathLike


def write_or_return(
    data: bytes, path_or_fobj: PathType | BufferedWriter | None = None
) -> bytes | None:
    # assume path_or_fobj is an open filelike
    if path_or_fobj is None:
        return data

    if isinstance(path_or_fobj, BufferedWriter):
        try:
            path_or_fobj.write(data)
        except TypeError as error:
            raise TypeError("File must be open for bytes writing") from error
    else:
        with open(path_or_fobj, "wb") as f:
            f.write(data)

    return None


# maybe temp location for FQ merge machinery
class FQPointKey(NamedTuple):
    expocode: str
    station: str
    cast: int
    sample: str


class FQProfileKey(NamedTuple):
    expocode: str
    station: str
    cast: int


class WHPIndxer:
    def __init__(self, obj: xr.Dataset) -> None:
        self.n_prof = pd.MultiIndex.from_arrays(
            [
                obj.expocode.data,
                obj.station.data,
                obj.cast.data,
            ],
            names=["expocode", "station", "cast"],
        )
        self.n_level = []
        for _, prof in obj.groupby("N_PROF", squeeze=False):
            data = prof.sample.squeeze("N_PROF").data
            self.n_level.append(pd.Index(data[data != ""]))

    def __getitem__(self, key: FQProfileKey | FQPointKey):
        prof_idx = self.n_prof.get_loc((key.expocode, key.station, key.cast))
        if isinstance(key, FQPointKey):
            level_idx = self.n_level[prof_idx].get_loc(key.sample)
        else:
            level_idx = slice(None)

        return prof_idx, level_idx


NormalizedFQ = dict[FQProfileKey | FQPointKey, dict[str, str | float]]


def normalize_fq(fq: list[dict[str, str | float]], *, check_dupes=True) -> NormalizedFQ:
    normalized: NormalizedFQ = defaultdict(dict)
    key: FQProfileKey | FQPointKey
    for line in fq:
        line = line.copy()
        expocode = str(line.pop("EXPOCODE"))
        station = str(line.pop("STNNBR"))
        cast = int(line.pop("CASTNO"))
        try:
            sample = str(line.pop("SAMPNO"))
        except KeyError:
            key = FQProfileKey(expocode, station, cast)
        else:
            key = FQPointKey(expocode, station, cast, sample)

        if check_dupes is True:
            shared_keys = normalized[key].keys() & line.keys()
            if len(shared_keys) != 0:
                raise ValueError(f"Duplicate input data found: {key}")

        normalized[key].update(line)

    return normalized


def fq_get_precisions(fq: NormalizedFQ) -> dict[str, int]:
    collect: dict[str, list[str]] = defaultdict(list)
    for value in fq.values():
        for param, data in value.items():
            if isinstance(data, str):
                collect[param].append(data)

    return {
        param: extract_numeric_precisions(data).item()
        for param, data in collect.items()
    }


FTypeOptions = Literal["cf", "exchange", "coards", "woce"]


@xr.register_dataset_accessor("cchdo")
class CCHDOAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def to_mat(self, fname):
        """Experimental Matlab .mat data file generator.

        The support for netCDF files in Matlab is really bad.
        Matlab also has no built in support for the standards
        we are trying to follow (CF, ACDD), the most egregious
        lack of support is how to deal with times in netCDF files.
        This was an attempt to make a mat file which takes
        care of some of the things matlab won't do for you.
        It requires scipy to function.

        The file it produces is in no way stable.
        """
        try:
            from scipy.io import savemat as scipy_savemat  # noqa
        except ImportError as error:
            raise ImportError("scipy is required for mat file saving") from error

        mat_dict = {}
        data = self._obj.to_dict()

        # flatten
        for coord, value in data["coords"].items():
            del value["dims"]
            mat_dict[coord] = value
        for param, value in data["data_vars"].items():
            del value["dims"]
            mat_dict[param] = value

        # cleanups for matlab users
        def to_matdate(dt):
            if dt is None:
                return "NaT"
            return dt.strftime("%d-%b-%Y %H:%M:%S")

        def dt_list_to_str_list(dtl):
            return list(map(to_matdate, dtl))

        for _, value in mat_dict.items():
            if value.get("attrs", {}).get("standard_name") == "time":
                # the case of list of lists is bottle closure times, which is a sparse array
                if any(isinstance(v, list) for v in value["data"]):
                    value["data"] = list(map(dt_list_to_str_list, value["data"]))
                else:
                    value["data"] = dt_list_to_str_list(value["data"])

            if "status_flag" in value.get("attrs", {}).get("standard_name", ""):
                value["data"] = np.nan_to_num(value["data"], nan=9)

        scipy_savemat(fname, mat_dict)

    def to_coards(self, path=None):
        from .legacy.coards import to_coards

        return write_or_return(to_coards(self._obj), path)

    def to_woce(self, path=None):
        from .legacy.woce import to_woce

        return write_or_return(to_woce(self._obj), path)

    def to_sum(self, path=None):
        """NetCDF to WOCE sumfile maker.

        This is missing some information that is not included anymore (wire out, height above bottom).
        It is especially lacking in including woce parameter IDs
        """
        COMMENTS = "CCHDO SumFile"  # TODO is there a better way?
        SUM_COLUMN_HEADERS_1 = [
            "SHIP/CRS",
            "WOCE",
            "",
            "",
            "CAST",
            "",
            "UTC",
            "EVENT",
            "",
            "POSITION",
            "",
            "UNC",
            "HT ABOVE",
            "WIRE",
            "MAX",
            "NO. OF",
            "",
            "",
        ]
        SUM_COLUMN_HEADERS_2 = [
            "EXPOCODE",
            "SECT",
            "STNNBR",
            "CASTNO",
            "TYPE",
            "DATE",
            "TIME",
            "CODE",
            "LATITUDE",
            "LONGITUDE",
            "NAV",
            "DEPTH",
            "BOTTOM",
            "OUT",
            "PRESS",
            "BOTTLES",
            "PARAMETERS",
            "COMMENTS",
        ]

        SUM_COL_JUSTIFICATION = [  # note that python calls this "align"
            "<",  # expo
            "<",  # woce line
            ">",  # station
            ">",  # cast
            ">",  # type
            "<",  # date
            ">",  # time
            ">",  # EVENT (guess at justification)
            "<",  # position 1
            "<",  # position 2
            "<",  # NAV (guess at justification)
            ">",  # depth
            ">",  # height
            ">",  # wire out
            ">",  # pres
            ">",  # no bottles
            "<",  # params
            "<",  # comments
        ]

        def sum_lat(deg_float):
            deg = abs(int(deg_float))
            if deg_float >= 0:
                hem = "N"
                dec = deg_float % 1
            else:
                hem = "S"
                dec = -(deg_float % -1)

            mins = 60 * dec

            return f"{deg:>2d} {mins:05.2f} {hem}"

        def sum_lon(deg_float):
            deg = abs(int(deg_float))
            if deg_float >= 0:
                hem = "E"
                dec = deg_float % 1
            else:
                hem = "W"
                dec = -(deg_float % -1)

            mins = 60 * dec

            return f"{deg:>3d} {mins:05.2f} {hem}"

        col_widths = [len(s) for s in SUM_COLUMN_HEADERS_1]
        col_widths = [max(x, len(s)) for x, s in zip(col_widths, SUM_COLUMN_HEADERS_2)]

        sum_rows = []
        for _, prof in self._obj.groupby("N_PROF", squeeze=False):
            prof = prof.squeeze("N_PROF")
            dt = pd.to_datetime(prof.time.values)

            sect_id = ""
            sect_ids = prof.filter_by_attrs(whp_name="SECT_ID")
            for _, ids in sect_ids.items():
                sect_id = str(ids.values)
                break

            depth = ""
            depths = prof.filter_by_attrs(whp_name="DEPTH")
            for _, meters in depths.items():
                depth = f"{meters.values:.0f}"
                if depth == "nan":
                    depth = ""
                break

            no_of_bottles = (
                f"{sum(prof.sample.values!='')}" if prof.profile_type == "B" else ""
            )

            row = [""] * len(col_widths)
            row[0] = str(
                prof.expocode.values
            )  # TODO? Limit to 12 chars as per 3.3.1 of woce manual
            row[1] = sect_id  # TODO? Maybe also limit to 12 chars?
            row[2] = str(prof.station.values)
            row[3] = str(prof.cast.values)
            row[4] = "ROS"
            row[5] = dt.strftime("%m%d%y")
            row[6] = dt.strftime("%H%M")
            row[7] = "BO"
            row[8] = sum_lat(prof.latitude.values)
            row[9] = sum_lon(prof.longitude.values)
            row[10] = "GPS"
            row[11] = depth
            row[12] = ""  # height above "BOTTOM"
            row[13] = ""  # "WIRE" out
            row[14] = f"{max(prof.pressure.values):.0f}"
            row[15] = no_of_bottles
            row[16] = (
                ""  # "PARAMS" we have this info... needs to be calculated on a per profile basis though...
            )
            row[17] = ""  # "COMMENTS"
            sum_rows.append(row)
            col_widths = [max(x, len(s)) for x, s in zip(col_widths, row)]

        formats = []
        for width, align in zip(col_widths, SUM_COL_JUSTIFICATION):
            formats.append("{: " + align + str(width) + "}")
        format_str = " ".join(formats)

        HEADERS_1 = format_str.format(*SUM_COLUMN_HEADERS_1)
        HEADERS_2 = format_str.format(*SUM_COLUMN_HEADERS_2)
        SEP_LINE = "-" * (sum(col_widths) + len(col_widths))
        SUM_ROWS = []
        for row in sum_rows:
            SUM_ROWS.append(format_str.format(*row))

        sum_file = "\n".join(
            [COMMENTS, HEADERS_1, HEADERS_2, SEP_LINE, *SUM_ROWS]
        ).encode("ascii")
        return write_or_return(sum_file, path)

    @property
    def __geo_interface__(self):
        """The station positions as a MultiPoint geo interface.

        See https://gist.github.com/sgillies/2217756
        """
        ds = self._obj
        coords = np.column_stack((ds.longitude, ds.latitude))

        return {"type": "MultiPoint", "coordinates": coords.tolist()}

    @property
    def track(self):
        """A dict which can be dumped to json which conforms to the expected structure for the CCHDO website."""
        geo = self.__geo_interface__
        if len(geo["coordinates"]) == 1:
            # Website only supports LineString which must contain at least 2 points
            # They can be the same point though
            geo["coordinates"].append(geo["coordinates"][0])

        geo["type"] = "LineString"
        return geo

    @staticmethod
    def _gen_fname(
        expocode: str,
        station: str,
        cast: int,
        profile_type: FileType,
        profile_count: int = 1,
        ftype: FTypeOptions = "cf",
    ) -> str:
        allowed_chars = set(f"._{string.ascii_letters}{string.digits}")

        ctd_one = "ctd.nc"
        ctd_many = "ctd.nc"
        bottle = "bottle.nc"

        if ftype == "exchange":
            ctd_one = "ct1.csv"
            ctd_many = "ct1.zip"
            bottle = "hy1.csv"

        if ftype == "coards":
            # internal zip filenames are done by the legacy writer
            ctd_one = "nc_ctd.zip"
            ctd_many = "nc_ctd.zip"
            bottle = "nc_hyd.zip"

        if ftype == "woce":
            # internal zip filenames are done by the legacy writer
            ctd_one = "ct.txt"
            ctd_many = "ct.zip"
            bottle = "hy.txt"

        if profile_type == FileType.BOTTLE:
            fname = f"{expocode}_{bottle}"
        elif profile_count > 1 or ftype in ("woce", "coards"):
            fname = f"{expocode}_{ctd_many}"
        else:
            fname = f"{expocode}_{station}_{cast:.0f}_{ctd_one}"

        for char in set(fname) - allowed_chars:
            fname = fname.replace(char, "_")

        return fname

    def gen_fname(self, ftype: FTypeOptions = "cf") -> str:
        """Generate a human friendly netCDF (or other output type) filename for this object."""
        expocode = np.atleast_1d(self._obj["expocode"])[0]
        station = np.atleast_1d(self._obj["station"])[0]
        cast = np.atleast_1d(self._obj["cast"])[0]

        profile_type = FileType(np.atleast_1d(self._obj["profile_type"])[0])
        profile_count = len(self._obj.get("N_PROF", []))

        return self._gen_fname(
            expocode, station, cast, profile_type, profile_count, ftype
        )

    def compact_profile(self):
        """Drop the trailing empty data from a profile.

        Because we use the incomplete multidimensional array representation of profiles
        there is often "wasted space" at the end of any profile that is not the longest one.
        This accessor drops that wasted space for xr.Dataset objects containing a single profile
        """
        if self._obj.sizes["N_PROF"] != 1:
            raise NotImplementedError(
                "Cannot compact Dataset with more than one profile"
            )
        return self._obj.isel(N_LEVELS=(self._obj.sample != "")[0])

    date_names = {WHPNames["DATE"], WHPNames["BTL_DATE"]}
    time_names = {WHPNames["TIME"], WHPNames["BTL_TIME"]}

    @property
    def file_type(self):
        # TODO profile_type is guaranteed to be present
        # TODO profile_type must have C or D as the value
        profile_type = self._obj.profile_type
        if not all_same(profile_type.values):
            raise NotImplementedError(
                "Unable to convert a mix of ctd and bottle (or unknown) dtypes"
            )

        if profile_type[0] == FileType.CTD.value:
            return FileType.CTD
        elif profile_type[0] == FileType.BOTTLE.value:
            return FileType.BOTTLE
        else:
            raise NotImplementedError("Unknown profile type encountered")

    @staticmethod
    def cchdo_c_format_precision(c_format: str) -> int | None:
        if not c_format.endswith("f"):
            return None

        f_format = re.compile(r"\.(\d+)f")
        if (match := f_format.search(c_format)) is not None:
            return int(match.group(1))
        return match

    def _make_params_units_line(
        self,
        params: dict[WHPName, xr.DataArray],
    ):
        plist = []
        ulist = []
        for param, dataarray in sorted(params.items()):
            if self.file_type == FileType.CTD and (
                param.scope != "sample" or param.nc_name == "sample"
            ):
                continue
            plist.append(param.full_whp_name)
            unit = param.whp_unit
            if unit is None:
                unit = ""

            ulist.append(unit)

            if (flag := dataarray.attrs.get(FLAG_NAME)) is not None:
                plist.append(flag.attrs["whp_name"])
                ulist.append("")

            if (error := dataarray.attrs.get(ERROR_NAME)) is not None:
                plist.append(error.attrs["whp_name"])
                ulist.append(unit)

        return ",".join(plist), ",".join(ulist)

    @staticmethod
    def _whpname_from_attrs(attrs) -> list[WHPName]:
        params = []
        param = attrs["whp_name"]
        unit = attrs.get("whp_unit")
        if isinstance(param, list):
            for combined in param:
                params.append(WHPNames[(combined, unit)])
        else:
            try:
                error = WHPNames[(param, unit)]
                if error.error_col:
                    return []
            except KeyError:
                pass
            params.append(WHPNames[(param, unit)])
        return params

    def _make_ctd_headers(self, params) -> list[str]:
        headers = {}
        for param, da in params.items():
            if param.scope != "profile":
                continue

            date_or_time = None
            if param in self.date_names:
                date_or_time = "date"
                value = da.dt.strftime("%Y%m%d").to_numpy()[0]
            elif param in self.time_names:
                date_or_time = "time"
                value = da.dt.round("min").dt.strftime("%H%M").to_numpy()[0]
            else:
                try:
                    data = da.values[0].item()
                except AttributeError:
                    data = da.values[0]

                numeric_precision_override = self.cchdo_c_format_precision(
                    da.attrs.get("C_format", "")
                )
                value = param.strfex(
                    data,
                    date_or_time=date_or_time,
                    numeric_precision_override=numeric_precision_override,
                )
            headers[param.whp_name] = value.strip()
        return [
            f"NUMBER_HEADERS = {len(headers) + 1}",
            *[f"{key} = {value}" for key, value in headers.items()],
        ]

    def _make_data_block(self, params: dict[WHPName, xr.DataArray]) -> list[str]:
        # TODO N_PROF is guaranteed
        valid_levels = params[WHPNames["SAMPNO"]] != ""
        data_block = []
        for param, da in sorted(params.items()):
            if self.file_type == FileType.CTD and (
                param.scope != "sample" or param.nc_name == "sample"
            ):
                continue
            date_or_time: Literal["date", "time"] | None = None
            # TODO, deal with missing time in BTL_DATE
            if param in self.date_names:
                date_or_time = "date"
                values = da[valid_levels].dt.strftime("%Y%m%d").to_numpy().tolist()
            elif param in self.time_names:
                date_or_time = "time"
                values = (
                    da[valid_levels]
                    .dt.round("min")
                    .dt.strftime("%H%M")
                    .to_numpy()
                    .tolist()
                )
            else:
                if da.dtype.char == "m":
                    nat_mask = np.isnat(da)
                    data_t = da.values.astype("timedelta64[s]").astype("float64")
                    data_t[nat_mask] = np.nan
                    data = np.nditer(data_t)
                else:
                    data = np.nditer(da[valid_levels], flags=["refs_ok"])
                numeric_precision_override = self.cchdo_c_format_precision(
                    da.attrs.get("C_format", "")
                )
                values = [
                    param.strfex(
                        v,
                        date_or_time=date_or_time,
                        numeric_precision_override=numeric_precision_override,
                    )
                    for v in data
                ]

            data_block.append(values)

            if (flags := da.attrs.get(FLAG_NAME)) is not None:
                data = np.nditer(flags[valid_levels])
                flag = [param.strfex(v, flag=True) for v in data]
                data_block.append(flag)

            if (errors := da.attrs.get(ERROR_NAME)) is not None:
                data = np.nditer(errors[valid_levels])
                numeric_precision_override = self.cchdo_c_format_precision(
                    da.attrs.get("C_format", "")
                )
                error = [
                    param.strfex(
                        v,
                        date_or_time=date_or_time,
                        numeric_precision_override=numeric_precision_override,
                    )
                    for v in data
                ]
                data_block.append(error)
        return data_block

    def _get_comments(self):
        output = []
        if len(comments := self._obj.attrs.get("comments", "")) > 0:
            for comment_line in comments.splitlines():
                output.append(f"#{comment_line}")
        return output

    def to_whp_columns(self, compact=False) -> dict[WHPName, xr.DataArray]:
        # collect all the Exchange variables
        # TODO, all things that appear in an exchange file, must have WHP name
        ds = flatten_cdom_coordinate(self._obj)

        ds = ds.reset_coords(
            [
                "expocode",
                "station",
                "cast",
                "sample",
                "time",
                "latitude",
                "longitude",
                "pressure",
            ]
        )

        ds = ds.stack(ex=("N_PROF", "N_LEVELS"))
        if compact:
            ds = ds.isel(ex=(ds.sample != ""))

        exchange_vars = ds.filter_by_attrs(whp_name=lambda name: name is not None)
        params: dict[WHPName, xr.DataArray] = {}
        for var in exchange_vars.values():
            whp_params = self._whpname_from_attrs(var.attrs)
            for param in whp_params:
                params[param] = var

            ancillary_vars_attr = var.attrs.get("ancillary_variables")
            if ancillary_vars_attr is None:
                continue

            # CF says these need to be space seperated
            ancillary_vars = ancillary_vars_attr.split(" ")
            for ancillary_var in ancillary_vars:
                ancillary = ds[ancillary_var]

                standard_name = ancillary.attrs.get("standard_name")
                if standard_name is None and ancillary.attrs.get("whp_name") is None:
                    # TODO maybe raise...
                    continue

                # currently there are three types of ancillary: flags, errors, and analytical temps (e.g. for pH)
                if standard_name == "temperature_of_analysis_of_sea_water":
                    # this needs to get treated like a param
                    for param in self._whpname_from_attrs(ancillary.attrs):
                        params[param] = ancillary

                elif standard_name == "status_flag":
                    for param in whp_params:
                        ancillary.attrs["whp_name"] = f"{param.full_whp_name}_FLAG_W"
                        params[param].attrs[FLAG_NAME] = ancillary

                # TODO find a way to test this
                try:
                    error_param = WHPNames[
                        (
                            ancillary.attrs.get("whp_name"),
                            ancillary.attrs.get("whp_unit"),
                        )
                    ]
                    if error_param.error_col:
                        ancillary.attrs["whp_name"] = error_param.full_error_name
                        params[param].attrs[ERROR_NAME] = ancillary
                except KeyError:
                    pass

        return params

    def to_exchange(self, path=None):
        """Convert a CCHDO CF netCDF dataset to exchange."""
        # all of the todo comments are for documenting/writing validators
        output_files = {}
        if self.file_type == FileType.CTD:
            for _, ds1 in self._obj.groupby("N_PROF", squeeze=False):
                fname = ds1.cchdo.gen_fname(ftype="exchange")

                output = []
                output.append(f"CTD,{datetime.now(timezone.utc):%Y%m%d}CCHHYDRO")
                output.extend(self._get_comments())

                params = ds1.cchdo.to_whp_columns()
                output.extend(self._make_ctd_headers(params))
                output.extend(self._make_params_units_line(params))

                data_block = self._make_data_block(params)
                for row in zip(*data_block):
                    output.append(",".join(str(cell) for cell in row))

                output.append("END_DATA\n")
                output_files[fname] = "\n".join(output).encode("utf8")

        if self.file_type == FileType.BOTTLE:
            fname = self._obj.cchdo.gen_fname(ftype="exchange")

            output = []
            output.append(f"BOTTLE,{datetime.now(timezone.utc):%Y%m%d}CCHHYDRO")
            output.extend(self._get_comments())
            params = self._obj.cchdo.to_whp_columns()

            # add the params and units line
            output.extend(self._make_params_units_line(params))

            data_block = self._make_data_block(params)

            for row in zip(*data_block):
                output.append(",".join(str(cell) for cell in row))

            output.append("END_DATA\n")
            output_files[fname] = "\n".join(output).encode("utf8")

        if len(output_files) == 1:
            return write_or_return(next(iter(output_files.values())), path)
        output_zip = BytesIO()
        with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zipfile:
            for fname, data in output_files.items():
                zipfile.writestr(fname, data)

        output_zip.seek(0)
        return write_or_return(output_zip.read(), path)

    # Until I figure out how to use the pandas machinery (or the explict index project of xarray pays off)
    # I will use a "custom" indexer here to index into the variables
    # This will rely on the N_PROF and N_LEVELS (with extra at some point)
    # * N_PROF will be indexed with (expocode, station, cast)
    # * N_LEVELS will be subindexd with (sample)
    def merge_fq(self, fq: list[dict[str, str | float]], *, check_flags=True):
        # TODOs...
        # * (default True) restrict to open "slots" of non flag 9s
        # * Update history attribute...
        now = datetime.now(timezone.utc)
        new_obj = self._obj.copy(deep=True)
        new_obj = flatten_cdom_coordinate(new_obj)
        idxer = WHPIndxer(new_obj)

        normalized_fq = normalize_fq(fq)
        input_precisions = fq_get_precisions(normalized_fq)
        idxes = {key: idxer[key] for key in normalized_fq}
        # invert keys and indexes?
        inverted: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for key, fq_values in normalized_fq.items():
            for param, value in fq_values.items():
                idx = idxes[key]
                inverted[param]["profs"].append(idx[0])
                inverted[param]["levels"].append(idx[1])
                inverted[param]["values"].append(value)

        for param, values in inverted.items():
            whpname = WHPNames[param]
            if whpname.error_col:
                col_ref = new_obj[whpname.nc_name_error]
            elif whpname.flag_col:
                col_ref = new_obj[whpname.nc_name_flag]
            else:
                col_ref = new_obj[whpname.full_nc_name]

            col_ref.values[values["profs"], values["levels"]] = values["values"]

            col_ref.attrs["date_modified"] = now.isoformat(timespec="seconds")

            if (
                param in input_precisions
                and whpname.dtype == "decimal"
                and not whpname.flag_col
            ):
                new_c_format = f"%.{input_precisions[param]}f"
                new_c_format_source = "input_file"
                if (
                    col_ref.attrs.get("C_format") != new_c_format
                    or col_ref.attrs.get("C_format_source") != new_c_format_source
                ):
                    col_ref.attrs["C_format"] = new_c_format
                    col_ref.attrs["C_format_source"] = new_c_format_source
                    col_ref.attrs["date_metadata_modified"] = now.isoformat(
                        timespec="seconds"
                    )
        new_obj = add_cdom_coordinate(new_obj)
        if check_flags:
            _check_flags(new_obj)
        return new_obj
