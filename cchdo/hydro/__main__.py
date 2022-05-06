import logging
from pathlib import Path
from multiprocessing import Pool
from tempfile import TemporaryDirectory
import shutil
from html import escape
import json

import click
from rich.logging import RichHandler
from rich.progress import track

log = logging.getLogger(__name__)


def setup_logging(level):
    FORMAT = "%(funcName)s: %(message)s"
    log_handler = RichHandler(level=level)
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[log_handler]
    )
    logging.captureWarnings(True)


@click.group()
def convert():
    ...


@convert.command()
@click.argument("exchange_path")
@click.argument("out_path")
@click.argument("precision_source", default="file")
@click.option("--check-flag/--no-check-flag", default=True)
def convert_exchange(exchange_path, out_path, check_flag, precision_source):
    setup_logging("DEBUG")
    log.info("Loading read_exchange")
    from .exchange import read_exchange

    checks = {"flags": check_flag}

    ex = read_exchange(exchange_path, checks=checks, precision_source=precision_source)
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
        log.debug(f"Using cached file {file_dest}")
        return file_dest

    f_body = s.get(f"https://cchdo.ucsd.edu{file['file_path']}")
    if f_body.status_code == codes.ok:
        file_dest.write_bytes(f_body.content)
        log.debug(f"Written to {file_dest}")

    return file_dest


from . import __main_helpers as mh


@status.command()
@click.argument("dtype")
@click.argument("out_dir")
@click.option("--dump-unknown-params", is_flag=True)
@click.option("-v", "--verbose", count=True)
def status_exchange(dtype, out_dir, dump_unknown_params, verbose):
    """Generate a bottle conversion status for all ex files of type type in the CCHDO Dataset"""
    from cchdo.hydro._version import version as hydro_version  # type: ignore
    from cchdo.params import _version as params_version  # type: ignore

    if verbose == 0:
        setup_logging("CRITICAL")
    if verbose == 1:
        setup_logging("INFO")
    if verbose >= 2:
        setup_logging("DEBUG")

    out_path = Path(out_dir)
    cruises, files = cchdo_loader(dtype)
    file_paths = []
    for file in track(files, description="Loading data files"):
        file_paths.append((cached_file_loader(file), file))

    results = []
    all_unknown_params = {}
    with TemporaryDirectory() as temp_dir:
        with Pool() as pool:
            for result in track(
                pool.imap_unordered(
                    mh.p_file, [(temp_dir, f[0], f[1]) for f in file_paths]
                ),
                total=len(file_paths),
                description="Converting ex to netCDF",
            ):
                status, path_or_err, metadata, unknown_params = result
                if status == 200:
                    log.info(f"Processed: {metadata['file_name']}")
                else:
                    log.error(f"Failed: {metadata['file_name']} {path_or_err}")
                results.append(result)

        out_path.mkdir(parents=True, exist_ok=True)
        nc_path = out_path / "nc"
        nc_path.mkdir(exist_ok=True)
        success_len = len(list(filter(lambda x: x[0] == 200, results)))
        success_str = f"Converted {success_len} of {len(results)} ({success_len/len(results) * 100}%)"
        with (out_path / f"index_{dtype}.html").open("w") as f:
            f.write(
                f"""<html>
            <head>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
            </head>
            <body>
            <div class="container-fluid">
            <table class="table"><thead>
            <h2>Versions</h2>
            cchdo.hydro: {hydro_version}</br>
            cchdo.params: {params_version.version}</br>
            stats: {success_str}
            <h2>Files</h2>
            <tr>
            <th>Cruise(s)</th>
            <th>File ID</th>
            <th>File Download</th>
            <th>NetCDF File</th>
            </tr></thead><tbody>"""
            )
            for result in sorted(results, key=lambda x: x[2]["id"]):
                status, path_or_err, metadata, unknown_params = result
                try:
                    expos = [cruises[c]["expocode"] for c in metadata["cruises"]]
                    crs = ", ".join(
                        [
                            f"<a href='https://cchdo.ucsd.edu/cruise/{ex}'>{ex}</a>"
                            for ex in expos
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
                    if len(unknown_params) > 0:
                        all_unknown_params[expos[0]] = unknown_params
                    error = escape(path_or_err)
                    f.write(
                        f"""<tr class='table-warning'>
                    <td>{crs}</td>
                    <td>{file_id}</td>
                    <td><a href="https://cchdo.ucsd.edu/data/{file_id}/{fn}">{fn}</a></td>"""
                    )

                    if "ExchangeDataFlagPairError" in path_or_err:
                        f.write(
                            f"""<td><pre style="width: fit-content"><code>{error}</code></pre></td>"""
                        )
                    else:
                        f.write(f"""<td>{error}</td>""")

                    f.write("""</tr>""")
            f.write("</tbody></table></div></body></html>")

    log.info(success_str)
    if dump_unknown_params:
        with open(out_path / f"unknown_params_{dtype}.json", "w") as f:
            json.dump(all_unknown_params, f)


cli = click.CommandCollection(sources=[convert, status])


if __name__ == "__main__":
    cli()
