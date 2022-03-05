import logging
from pathlib import Path
from multiprocessing import Pool
from tempfile import TemporaryDirectory
import shutil

import click
from rich.logging import RichHandler
from rich.progress import track


FORMAT = "%(funcName)s: %(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger(__name__)


@click.group()
def convert():
    ...


@convert.command()
@click.argument("exchange_path")
@click.argument("out_path")
def convert_exchnage(exchange_path, out_path):
    log.info("Loading read_exchange")
    from .exchange.two_pass import read_exchange

    ex = read_exchange(exchange_path)
    log.info("Saving to netCDF")
    ex.to_netcdf(out_path)
    log.info("Done :)")


@click.group()
def status():
    ...


def cchdo_loader(dtype):
    from cchdo.auth.session import session as s  # type: ignore

    log.info("Loading Cruise Metadata")
    cruises = s.get("https://cchdo.ucsd.edu/api/v1/cruise/all").json()
    log.info("Loading Cruise File")
    files = s.get("https://cchdo.ucsd.edu/api/v1/file/all").json()

    def file_filter(file):
        return (
            file["data_type"] == dtype
            and file["role"] == "dataset"
            and file["data_format"] == "exchange"
        )

    return {c["id"]: c for c in cruises}, list(filter(file_filter, files))


def cached_file_loader(file):
    from requests import codes
    from cchdo.auth.session import session as s
    from . import _hydro_appdirs

    cache_dir = Path(_hydro_appdirs.user_cache_dir) / "convert"
    cache_dir.mkdir(parents=True, exist_ok=True)

    file_dest = cache_dir / file["file_hash"]

    if file_dest.exists() and file_dest.stat().st_size == file["file_size"]:
        log.info(f"Using cached file {file_dest}")
        return file_dest

    f_body = s.get(f"https://cchdo.ucsd.edu{file['file_path']}")
    if f_body.status_code == codes.ok:
        file_dest.write_bytes(f_body.content)
        log.info(f"Written to {file_dest}")

    return file_dest


from . import __main_helpers as mh


@status.command()
@click.argument("dtype")
@click.argument("out_dir")
def status_exchange(dtype, out_dir):
    """Generate a bottle conversion status for all ex files of type type in the CCHDO Dataset"""
    from cchdo.hydro._version import version as hydro_version  # type: ignore
    from cchdo.params import _version as params_version  # type: ignore

    out_path = Path(out_dir)
    cruises, files = cchdo_loader(dtype)
    file_paths = []
    for file in track(files, description="Loading data files"):
        file_paths.append((cached_file_loader(file), file))

    results = []
    with TemporaryDirectory() as temp_dir:
        with Pool() as pool:
            for result in track(
                pool.imap_unordered(
                    mh.p_file, [(temp_dir, f[0], f[1]) for f in file_paths]
                ),
                total=len(file_paths),
                description="Converting ex to netCDF",
            ):
                status, path_or_err, metadata = result
                if status == 200:
                    log.info(f"Processed: {metadata['file_name']}")
                else:
                    log.error(f"Failed: {metadata['file_name']} {path_or_err}")
                results.append(result)

        out_path.mkdir(parents=True, exist_ok=True)
        nc_path = out_path / "nc"
        nc_path.mkdir(exist_ok=True)
        with (out_path / f"index_{dtype}.html").open("w") as f:
            f.write(
                f"""<html>
            <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
            </head>
            <body><table class="table"><thead>
            <h2>Versions</h2>
            cchdo.hydro: {hydro_version}</br>
            cchdo.params: {params_version.version}
            <h2>Files</h2>
            <tr>
            <th>Cruise(s)</th>
            <th>File ID</th>
            <th>File Download</th>
            <th>NetCDF File</th>
            </tr></thead><tbody>"""
            )
            for result in sorted(results, key=lambda x: x[-1]["id"]):
                status, path_or_err, metadata = result
                try:
                    crs = [cruises[c]["expocode"] for c in metadata["cruises"]]
                    crs = ", ".join(
                        [
                            f"<a href='https://cchdo.ucsd.edu/cruise/{ex}'>{ex}</a>"
                            for ex in crs
                        ]
                    )
                except KeyError:
                    continue
                except IndexError:
                    log.critical(metadata["cruises"])
                    raise
                fn = metadata["file_name"]
                file_id = metadata["id"]
                if status == 200:
                    tmp_nc = Path(path_or_err)
                    res_nc = nc_path / tmp_nc.name
                    shutil.copy(tmp_nc, res_nc)
                    f.write(
                        f"""<tr class='table-success'>
                            <td>{crs}</td>
                            <td>{file_id}</td>
                            <td><a href="https://cchdo.ucsd.edu/data/{file_id}/{fn}">{fn}</a></td>
                            <td><a href="nc/{res_nc.name}">{res_nc.name}</a></td>
                            </tr>"""
                    )
                else:
                    f.write(
                        f"""<tr class='table-warning'>
                    <td>{crs}</td>
                    <td>{file_id}</td>
                    <td><a href="https://cchdo.ucsd.edu/data/{file_id}/{fn}">{fn}</a></td>
                    <td>Error:{path_or_err}</td>
                    </tr>"""
                    )
            f.write("</tbody></table></body></html>")

    success_len = len(list(filter(lambda x: x[0] == 200, results)))
    log.info(
        f"Converted {success_len} of {len(results)} ({success_len/len(results) * 100}%)"
    )


cli = click.CommandCollection(sources=[convert, status])


if __name__ == "__main__":
    cli()
