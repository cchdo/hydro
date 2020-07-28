import xarray as xr
import pandas as pd


@xr.register_dataset_accessor("woce")
class WoceAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def sum_file(self, path=None):
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

            min = 60 * (dec / (10 ** dec_len))

            return f"{deg} {min:05.2f} {hem}"

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

            min = 60 * (dec / (10 ** dec_len))

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
                depth = str(meters.values)
                break

            row = [""] * len(col_widths)
            row[0] = str(prof.expocode.values)
            row[1] = sect_id
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
            row[14] = ""  # max "PRESS"
            row[15] = ""  # TODO number of bottle trips... str(len(cast.samples))
            row[16] = ""  # "PARAMS"
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
            with open(path, "w") as f:
                f.write(sum_file)
                return

        return sum_file
