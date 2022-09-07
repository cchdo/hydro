from cchdo.params import WHPNames

print(
    """.. list-table:: Variables In ERDDAP
   :header-rows: 1

   * - netcdf varname
     - Units
     - In ERDDAP"""
)

seen = set()
for name in sorted(WHPNames.values()):
    if name in seen:
        continue
    seen.add(name)
    yes = ":bdg-success:`yes`"
    no = ":bdg-danger:`no`"

    print(
        f"""   * - ``{name.nc_name}``
     - {name.cf_unit}
     - {yes if name.in_erddap else no}"""
    )
