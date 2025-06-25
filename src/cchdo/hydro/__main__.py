import json
import logging
import shutil
from collections import Counter
from hashlib import sha256
from html import escape
from multiprocessing import Pool
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import click
import numpy as np
import xarray as xr
from rich.logging import RichHandler
from rich.progress import track

from . import __main_helpers as mh

log = logging.getLogger(__name__)

from . import __version__


def setup_logging(level):
    FORMAT = "%(funcName)s: %(message)s"
    log_handler = RichHandler(level=level)
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[log_handler]
    )
    logging.captureWarnings(True)


@click.group()
def convert(): ...


def _comment_loader(str_or_path: str) -> str:
    if str_or_path.startswith("@"):
        with open(str_or_path.removeprefix("@")) as f:
            return f.read()
    return str_or_path


PrecisionSouceType = click.Choice(["file", "database"], case_sensitive=False)


@convert.command()
@click.argument("exchange_path")
@click.argument("out_path")
@click.option("--precision_source", default="file", type=PrecisionSouceType)
@click.option("--check-flag/--no-check-flag", default=True)
@click.option(
    "--comments",
    default=None,
    type=str,
    help="either a comment string or file path prefixed with @ (e.g. @README.txt)",
)
def convert_exchange(exchange_path, out_path, check_flag, precision_source, comments):
    setup_logging("DEBUG")
    log.info("Loading read_exchange")
    from .exchange import read_exchange

    checks = {"flags": check_flag}

    ex = read_exchange(exchange_path, checks=checks, precision_source=precision_source)
    log.info("Saving to netCDF")
    if comments is not None:
        comments_contents = _comment_loader(comments)
        ex.attrs["comments"] = comments_contents
    ex.to_netcdf(out_path)
    log.info("Done :)")


@convert.command()
@click.argument("csv_path")
@click.argument("out_path")
@click.option(
    "--ftype",
    default="B",
    type=click.Choice(["B", "C"], case_sensitive=False),
    help="B for Bottle, C for CTD",
)
@click.option("--precision_source", default="file", type=PrecisionSouceType)
@click.option("--check-flag/--no-check-flag", default=True)
@click.option(
    "--comments",
    default=None,
    type=str,
    help="either a comment string or file path prefixed with @ (e.g. @README.txt)",
)
def convert_csv(csv_path, out_path, ftype, check_flag, precision_source, comments):
    setup_logging("DEBUG")
    log.info("Loading read_exchange")
    from .exchange import read_csv

    checks = {"flags": check_flag}

    ex = read_csv(
        csv_path, ftype=ftype, checks=checks, precision_source=precision_source
    )
    log.info("Saving to netCDF")
    if comments is not None:
        comments_contents = _comment_loader(comments)
        ex.attrs["comments"] = comments_contents
    ex.to_netcdf(out_path)
    log.info("Done :)")


@click.group()
def edit(): ...


@edit.command(hidden=True)
@click.argument("expocode")
@click.argument("dtype")
def edit_comments(expocode, dtype):
    from cchdo.auth.session import session as s

    log.info("Loading Cruise Metadata")
    cruises = {
        c["expocode"]: c
        for c in s.get("https://cchdo.ucsd.edu/api/v1/cruise/all").json()
    }
    try:
        cruise = cruises[expocode]
    except KeyError:
        log.error(f"{expocode} not found")
        raise

    cruise_file_ids = set(cruise["files"])

    log.info("Loading Cruise Files")
    files = {f["id"]: f for f in s.get("https://cchdo.ucsd.edu/api/v1/file/all").json()}
    extant_ids = cruise_file_ids & files.keys()
    edit_files = list(
        filter(
            lambda x: x["role"] == "dataset"
            and x["data_format"] == "cf_netcdf"
            and x["data_type"] == dtype,
            (files[id] for id in extant_ids),
        )
    )

    if len(edit_files) == 0:
        log.error("No files to edit")
        exit(1)
    if len(edit_files) > 1:
        log.error("Too many files to edit")
        exit(2)
    file = edit_files[0]
    with NamedTemporaryFile(suffix=".nc") as tf:
        tf.write(s.get(f"https://cchdo.ucsd.edu/{file['file_path']}").content)
        tf.flush()  # when 3.12, switch to delete_on_close=False
        ds = xr.load_dataset(tf.name)

    comments = click.edit(text=ds.attrs.get("comments", ""), extension=".txt")
    if comments is None:
        raise click.Abort

    old_file_id = file["id"]
    old_cruises = file["cruises"]
    new_file = file
    new_file["file_path"] = ""
    new_file["events"] = []
    new_file["submissions"] = []
    del new_file["cruises"]
    del new_file["id"]

    ds.attrs["comments"] = comments
    with TemporaryDirectory() as td:
        temp_nc = td + "/out.nc"
        ds.to_netcdf(temp_nc)
        new_file["file"] = mh.make_netcdf_file_json(temp_nc)
        new_file["file"]["name"] = new_file["file_name"]
        with open(temp_nc, "rb") as f:
            new_file["file_hash"] = sha256(f.read()).hexdigest()
            new_file["file_size"] = f.tell()

    patch = [{"op": "replace", "path": "/role", "value": "merged"}]

    replacement = s.post("https://cchdo.ucsd.edu/api/v1/file", json=new_file)
    print(replacement.json())

    id_ = replacement.json()["message"].split("/")[-1]
    for cruise in old_cruises:
        print(s.post(f"https://cchdo.ucsd.edu/api/v1/cruise/{cruise}/files/{id_}"))

    s.patch(f"https://cchdo.ucsd.edu/api/v1/file/{old_file_id}", json=patch)


@click.group()
def status(): ...


def cchdo_loader(dtype, dformat="exchange"):
    from cchdo.auth.session import session as s

    log.info("Loading Cruise Metadata")
    cruises = s.get("https://cchdo.ucsd.edu/api/v1/cruise/all").json()
    log.info("Loading Cruise File")
    files = s.get("https://cchdo.ucsd.edu/api/v1/file/all").json()

    def file_filter(file):
        return (
            file["data_type"] == dtype
            and file["role"] == "dataset"
            and file["data_format"] == dformat
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


def vars_with_value(ds: xr.Dataset) -> list[str]:
    vars_with_data: list[str] = []
    for name, var in ds.variables.items():
        if var.dtype.kind == "f":
            if np.any(np.isfinite(var)).item():
                vars_with_data.append(str(name))
        else:
            vars_with_data.append(str(name))
    return vars_with_data


@status.command()
@click.argument("dtype")
@click.argument("out_dir")
@click.option("--dump-unknown-params", is_flag=True)
@click.option("-v", "--verbose", count=True)
@click.option("--dump-data-counts", is_flag=True)
@click.option("--roundtrip", is_flag=True)
def status_exchange(
    dtype, out_dir, dump_unknown_params, verbose, dump_data_counts, roundtrip
):
    """Generate a bottle conversion status for all ex files of type type in the CCHDO Dataset."""
    from cchdo.hydro import __version__ as hydro_version
    from cchdo.params import __version__ as params_version

    if verbose == 0:
        setup_logging("CRITICAL")
    if verbose == 1:
        setup_logging("INFO")
    if verbose >= 2:
        setup_logging("DEBUG")

    out_path = Path(out_dir)
    cruises, files = cchdo_loader(dtype)
    file_paths = [
        (cached_file_loader(file), file)
        for file in track(files, description="Loading data files")
    ]

    results = []
    all_unknown_params = {}
    variables_with_data = []
    with TemporaryDirectory() as temp_dir:
        with Pool() as pool:
            for result in track(
                pool.imap_unordered(
                    mh.p_file, [(temp_dir, f[0], f[1], roundtrip) for f in file_paths]
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
        success_str = f"Converted {success_len} of {len(results)} ({success_len / len(results) * 100}%)"
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
            cchdo.params: {params_version}</br>
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
                    crs = "<span>Not Attached</span>"
                except IndexError:
                    log.critical(metadata["cruises"])
                    raise
                fn = metadata["file_name"]
                file_id = metadata["id"]
                if status == 200:
                    tmp_nc = Path(path_or_err)
                    res_nc = nc_path / tmp_nc.name

                    if dump_data_counts:
                        ds = xr.load_dataset(tmp_nc)
                        variables_with_data.extend(vars_with_value(ds))

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
    if dump_data_counts:
        with open(out_path / f"params_with_data_{dtype}.json", "w") as f:
            json.dump(Counter(variables_with_data), f)


@status.command()
@click.argument("out_dir")
@click.option("-v", "--verbose", count=True)
@click.option("--only-fail", is_flag=True)
def status_cf_derived(out_dir, verbose, only_fail):
    from cchdo.hydro import __version__ as hydro_version
    from cchdo.params import __version__ as params_version

    if verbose == 0:
        setup_logging("CRITICAL")
    if verbose == 1:
        setup_logging("INFO")
    if verbose >= 2:
        setup_logging("DEBUG")

    out_path = Path(out_dir)
    cruises, bottle_files = cchdo_loader("bottle", "cf_netcdf")
    _, ctd_files = cchdo_loader("ctd", "cf_netcdf")
    all_files = [*bottle_files, *ctd_files]
    file_paths = [
        (cached_file_loader(file), file)
        for file in track(all_files, description="Loading data files")
    ]

    results = []
    with TemporaryDirectory() as temp_dir:
        with Pool() as pool:
            for result in track(
                pool.imap_unordered(
                    mh.p_file_cf, [(temp_dir, f[0], f[1]) for f in file_paths]
                ),
                total=len(file_paths),
                description="Converting CF to Others",
            ):
                metadata, excahnge_ok, coards_ok, woce_ok, sum_ok = result
                log.info(
                    f"Processed: {metadata['file_name']}, {excahnge_ok}, {coards_ok}, {woce_ok}, {sum_ok}"
                )
                results.append(result)

        with (out_path / "index_cf.html").open("w") as f:
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
            cchdo.params: {params_version}</br>
            <h2>Files</h2>
            <tr>
            <th>Cruise(s)</th>
            <th>File ID</th>
            <th>File Download</th>
            <th>Exchange</th>
            <th>COARDS NC</th>
            <th>WOCE</th>
            <th>SUM</th>
            </tr></thead><tbody>"""
            )
            for result in sorted(results, key=lambda x: x[0]["id"]):
                metadata, excahnge_ok, coards_ok, woce_ok, sum_ok = result
                if only_fail and all([excahnge_ok, coards_ok, woce_ok, sum_ok]):
                    continue
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
                f.write(
                    f"""<tr>
                        <td>{crs}</td>
                        <td>{file_id}</td>
                        <td><a href="https://cchdo.ucsd.edu/data/{file_id}/{fn}">{fn}</a></td>
                        <td>{"✅" if excahnge_ok else "❌"}</td>
                        <td>{"✅" if coards_ok else "❌"}</td>
                        <td>{"✅" if woce_ok else "❌"}</td>
                        <td>{"✅" if sum_ok else "❌"}</td>
                        </tr>"""
                )
            f.write("</tbody></table></div></body></html>")


cli = click.version_option(__version__)(
    click.CommandCollection(sources=[convert, status, edit])
)


if __name__ == "__main__":
    cli()
