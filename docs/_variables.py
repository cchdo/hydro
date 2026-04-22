from pathlib import Path

from cchdo.hydro.core import dataarray_factory
from cchdo.params import WHPNames

base = Path("variables")
base.mkdir(exist_ok=True)

seen = set()
toc = []
for name in sorted(WHPNames.values()):
    if name in seen:
        continue
    seen.add(name)
    toc.append(name.full_nc_name)
    with (base / f"{name.full_nc_name}.md").open("w") as fo:
        da = dataarray_factory(name)
        fo.write(f"# ``{da.name}``\n")
        fo.write("## Properties\n")
        fo.write(f":dimensions: {', '.join(da.dims)}\n")
        fo.write(f":dtype: {da.dtype.name}\n")
        fo.write("## Default Attributes\n")
        fo.write("| Attribute | Value |\n|-----|-----|\n")
        fo.writelines(
            f"|``{attr}`` | ``{value}`` |\n" for attr, value in da.attrs.items()
        )
        fo.write("## Description\n")
        fo.write(f"{name.description}\n")

        if name.warning:
            fo.write(f":::{{warning}}\n{name.warning}\n:::")

with open("variables.md", "w") as fo:
    fo.write("# Variable List\n")
    fo.write(":::{toctree}\n")
    fo.writelines(f"variables/{entry}\n" for entry in toc)
    fo.write(":::\n")
