import logging

import click
from rich.logging import RichHandler

FORMAT = "%(funcName)s: %(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger(__name__)


@click.command()
@click.argument("exchange_path")
@click.argument("out_path")
def cli(exchange_path, out_path):
    log.info("Loading read_exchange")
    from .exchange import read_exchange

    ex = read_exchange(exchange_path)
    log.info("Convert to xr.Dataset")
    ex_xr = ex.to_xarray()
    log.info("Saving to netCDF")
    ex_xr.to_netcdf(out_path)


if __name__ == "__main__":
    cli()
