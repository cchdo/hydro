from typing import List, Dict, Optional, Union
import xarray as xr
import pandas as pd
import numpy as np
import string
import re

from cchdo.params import WHPNames, WHPName

from .exchange import FileType, all_same, check_flags as _check_flags


class CCHDOAccessorBase:
    """Class base for CCHDO accessors

    saves the xarray object to self._obj for all the subclasses
    """

    def __init__(self, xarray_obj: Union[xr.DataArray, xr.Dataset]):
        self._obj = xarray_obj


class MatlabAccessor(CCHDOAccessorBase):
    """Accessor containing the experimental matlab machinery"""

    def to_mat(self, fname):
        """Experimental Matlab .mat data file generator

        The support for netCDF files in Matlab is really bad.
        Matlab also has no built in support for the standards
        we are trying to follow (CF, ACDD), the most egregious
        lack of support is how to deal with times in netCDF files.
        This was an attempt to make a mat file which takes
        care of some of the things matlab won't do for you.
        It requires scipy to function.

        The file it produces is in no way stable.
        """
        from scipy.io import savemat as scipy_savemat

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


class WoceAccessor(CCHDOAccessorBase):
    """Accessor containing woce file output machinery"""

    def to_sum(self, path=None):
        """netCDF to WOCE sumfile maker

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

        def sum_lat(deg_str):
            deg_str = str(deg_str)
            deg, dec = deg_str.split(".")
            deg = int(deg)
            if deg > 0:
                hem = "N"
            else:
                hem = "S"
            deg = abs(deg)

            dec_len = len(dec)
            dec = int(dec)

            min = 60 * (dec / (10**dec_len))

            return f"{deg:>2d} {min:05.2f} {hem}"

        def sum_lon(deg_str):
            deg_str = str(deg_str)
            deg, dec = deg_str.split(".")
            deg = int(deg)
            if deg > 0:
                hem = "E"
            else:
                hem = "W"
            deg = abs(deg)

            dec_len = len(dec)
            dec = int(dec)

            min = 60 * (dec / (10**dec_len))

            return f"{deg:>3d} {min:05.2f} {hem}"

        col_widths = [len(s) for s in SUM_COLUMN_HEADERS_1]
        col_widths = [max(x, len(s)) for x, s in zip(col_widths, SUM_COLUMN_HEADERS_2)]

        sum_rows = []
        for _, prof in self._obj.groupby("N_PROF"):
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
            row[
                16
            ] = ""  # "PARAMS" we have this info... needs to be calculated on a per profile basis though...
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

        sum_file = "\n".join([COMMENTS, HEADERS_1, HEADERS_2, SEP_LINE, *SUM_ROWS])
        if path is not None:
            with open(path, "w", encoding="ascii") as f:
                f.write(sum_file)
                return

        return sum_file


class GeoAccessor(CCHDOAccessorBase):
    """Accessor providing geo_interface machinery"""

    @property
    def __geo_interface__(self):
        """The station positions as a MultiPoint geo interface

        See https://gist.github.com/sgillies/2217756
        """
        ds = self._obj
        coords = np.column_stack((ds.longitude, ds.latitude))

        return {"type": "MultiPoint", "coordinates": coords.tolist()}

    @property
    def track(self):
        """A dict which can be dumped to json which conforms to the expected
        structure for the CCHDO website
        """
        geo = self.__geo_interface__
        if len(geo["coordinates"]) == 1:
            # Website only supports LineString which must contain at least 2 points
            # They can be the same point though
            geo["coordinates"].append(geo["coordinates"][0])

        geo["type"] = "LineString"
        return geo


class MiscAccessor(CCHDOAccessorBase):
    """Accessor with misc functions that don't fit in some other category"""

    def gen_fname(self, ftype="cf") -> str:
        """Generate a human friendly netCDF filename for this object"""
        ds = self._obj
        allowed_chars = set(f"._{string.ascii_letters}{string.digits}")

        ctd_one = "ctd.nc"
        ctd_many = "ctd.nc"
        bottle = "bottle.nc"

        if ftype == "exchange":
            ctd_one = "ct1.csv"
            ctd_many = "ct1.zip"
            bottle = "hy1.csv"

        expocode = np.atleast_1d(ds["expocode"])[0]
        station = np.atleast_1d(ds["station"])[0]
        cast = np.atleast_1d(ds["cast"])[0]

        profile_type = np.atleast_1d(ds["profile_type"])[0]

        if profile_type == FileType.BOTTLE.value:
            fname = f"{expocode}_{bottle}"
        elif profile_type == FileType.CTD.value and len(ds.get("N_PROF", [])) > 1:
            fname = f"{expocode}_{ctd_many}"
        else:
            fname = f"{expocode}_{station}_{cast:.0f}_{ctd_one}"

        for char in set(fname) - allowed_chars:
            fname = fname.replace(char, "_")

        return fname


class ExchangeAccessor(CCHDOAccessorBase):
    """Class containing the to_exchange functionn"""

    @staticmethod
    def cchdo_c_format_precision(c_format: str) -> Optional[int]:
        if not c_format.endswith("f"):
            return None

        f_format = re.compile(r"\.(\d+)f")
        if (match := f_format.search(c_format)) is not None:
            return int(match.group(1))
        return match

    @staticmethod
    def _make_params_units_line(
        params: Dict[WHPName, xr.DataArray],
        flags: Dict[WHPName, xr.DataArray],
        errors: Dict[WHPName, xr.DataArray],
    ):
        plist = []
        ulist = []
        for param in params:
            plist.append(param.whp_name)
            unit = param.whp_unit
            if unit is None:
                unit = ""

            ulist.append(unit)

            if param in flags:
                plist.append(f"{param.whp_name}_FLAG_W")
                ulist.append("")

            if param in errors:
                if param.error_name is None:
                    raise ValueError(f"No error name for {param}")
                plist.append(param.error_name)
                ulist.append(unit)

        return ",".join(plist), ",".join(ulist)

    @staticmethod
    def _whpname_from_attrs(attrs) -> List[WHPName]:
        params = []
        param = attrs["whp_name"]
        unit = attrs.get("whp_unit")
        if isinstance(param, list):
            for combined in param:
                params.append(WHPNames[(combined, unit)])
        else:
            if (param, unit) in WHPNames.error_cols:
                return []
            params.append(WHPNames[(param, unit)])
        return params

    def to_exchange(self):
        """Convert a CCHDO CF netCDF dataset to exchange"""
        # all of the todo comments are for documenting/writing validators

        date_names = {WHPNames["DATE"], WHPNames["BTL_DATE"]}
        time_names = {WHPNames["TIME"], WHPNames["BTL_TIME"]}

        # TODO guarantee these coordinates
        ds = self._obj.reset_coords(
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
        ).stack(ex=("N_PROF", "N_LEVELS"))

        # TODO profile_type is guaranteed to be present
        # TODO profile_type must have C or D as the value
        profile_type = ds.profile_type
        if not all_same(profile_type.values):
            raise NotImplementedError(
                "Unable to convert a mix of ctd and bottle dtypes"
            )

        if profile_type[0] != FileType.BOTTLE.value:
            raise NotImplementedError("CTD conversion not implimentaed yet")

        output = []
        output.append("BOTTLE,date_initals")

        if len(comments := ds.attrs.get("comments", "")) > 0:
            for comment_line in comments.splitlines():
                output.append(f"#{comment_line}")

        # collect all the Exchange variables
        # TODO, all things that appear in an exchange file, must have WHP name
        exchange_vars = ds.filter_by_attrs(whp_name=lambda name: name is not None)
        params: Dict[WHPName, xr.DataArray] = {}
        flags = {}
        errors = {}
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

                if standard_name == "status_flag":
                    for param in whp_params:
                        flags[param] = ancillary

                # TODO find a way to test this
                if ancillary.attrs.get("whp_name") in WHPNames.error_cols:
                    for param in whp_params:
                        errors[param] = ancillary

        # add the params and units line
        output.extend(self._make_params_units_line(params, flags, errors))

        # TODO N_PROF is guaranteed
        # for _, prof in ds.groupby("N_PROF"):
        #    # TODO sample is empty/null string for... non samples
        #    valid_levels = prof.sample != ""
        valid_levels = params[WHPNames["SAMPNO"]] != ""
        data_block = []
        for param, da in params.items():
            date_or_time = None
            if param in date_names:
                date_or_time = "date"
                values = da[valid_levels].dt.strftime("%Y%m%d").to_numpy().tolist()
            elif param in time_names:
                date_or_time = "time"
                values = da[valid_levels].dt.strftime("%H%M").to_numpy().tolist()
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

            if param in flags:
                data = np.nditer(flags[param][valid_levels])
                flag = [param.strfex(v, flag=True) for v in data]
                data_block.append(flag)

            if param in errors:
                data = np.nditer(errors[param][valid_levels])
                numeric_precision_override = (
                    self.cchdo_c_format_precision(da.attrs.get("C_format", "")),
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

        for row in zip(*data_block):
            output.append(",".join(str(cell) for cell in row))

        output.append("END_DATA\n")

        return "\n".join(output)


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
        self.n_level = [
            pd.MultiIndex.from_arrays([prof.sample.data], names=["sample"])
            for _, prof in obj.groupby("N_PROF")
        ]

    def __getitem__(self, key):
        expocode = str(key.pop("EXPOCODE"))
        station = str(key.pop("STNNBR"))
        cast = int(key.pop("CASTNO"))
        sample = str(key.pop("SAMPNO"))
        prof_idx = self.n_prof.get_loc((expocode, station, cast))
        level_idx = self.n_level[prof_idx].get_loc((sample))

        return prof_idx, level_idx


class MergeFQAccessor(CCHDOAccessorBase):
    # Until I figure out how to use the pandas machinery (or the explict index project of xarray pays off)
    # I will use a "custom" indexer here to index into the variables
    # This will rely on the N_PROF and N_LEVELS (with extra at some point)
    # * N_PROF will be indexed with (expocode, station, cast)
    # * N_LEVELS will be subindexd with (sample)
    def merge_fq(self, fq, check_flags=True):
        new_obj = self._obj.copy(deep=True)
        idxer = WHPIndxer(new_obj)

        for line in fq:
            prof, level = idxer[line]
            for param, value in line.items():
                if param in WHPNames.error_cols:
                    whpname = WHPNames.error_cols[param]
                    new_obj[whpname.nc_name_error][prof, level] = value
                if param.endswith("_FLAG_W"):
                    whpname = WHPNames[param[:-7]]
                    new_obj[whpname.nc_name_flag][prof, level] = value
                else:
                    whpname = WHPNames[param]
                    new_obj[whpname.nc_name][prof, level] = value

        if check_flags:
            _check_flags(new_obj)
        return new_obj


class CCHDOAccessor(
    ExchangeAccessor,
    GeoAccessor,
    WoceAccessor,
    MatlabAccessor,
    MiscAccessor,
    MergeFQAccessor,
):
    """Collect all the accessors into a single class"""

    ...


xr.register_dataset_accessor("cchdo")(CCHDOAccessor)
