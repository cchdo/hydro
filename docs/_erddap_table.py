from cchdo.params import WHPNames
from cchdo.params import __version__ as params_version

with open("_erddap_table.md", "w") as fo:
    fo.write(
        f":::{{table}} Variables In ERDDAP as of cchdo parameters list version {params_version}\n"
    )
    fo.write("| netcdf varname | Units | In ERDDAP |\n")
    fo.write("| ------ | ------ | ------ |\n")

    seen = set()
    for name in sorted(WHPNames.values()):
        if name in seen:
            continue
        seen.add(name)
        yes = "{bdg-success}`yes`"
        no = "{bdg-danger}`no`"

        fo.write(
            f"|``{name.nc_name}`` | {name.cf_unit} | {yes if name.in_erddap else no} |\n"
        )
    fo.write(":::\n")
