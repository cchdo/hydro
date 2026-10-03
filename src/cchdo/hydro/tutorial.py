import io
import os
import typing
from collections.abc import Generator, Mapping
from zipfile import ZipFile

import requests

from . import _hydro_platformdirs

bottle_uri = "https://cchdo.ucsd.edu/search?q=a&download=exchange%2cbottle"
bottle_fname = "bottle_data.zip"


def _cache_dir():
    path = _hydro_platformdirs.user_cache_dir
    os.makedirs(path, exist_ok=True)
    return path


def load_cchdo_bottle_data():
    """Downloads some CCHDO data for playing with..."""
    path = os.path.join(_cache_dir(), bottle_fname)
    with requests.get(bottle_uri, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


class CCHDOBottleData(Mapping[str, io.BytesIO]):
    def __init__(self):
        self.path = os.path.join(_hydro_platformdirs.user_cache_dir, bottle_fname)
        try:
            with ZipFile(self.path) as f:
                self.files = f.namelist()
        except FileNotFoundError:
            load_cchdo_bottle_data()
            with ZipFile(self.path) as f:
                self.files = f.namelist()

    @typing.override
    def __len__(self) -> int:
        return len(self.files)

    @typing.override
    def __iter__(self) -> Generator[str]:
        yield from self.files

    @typing.override
    def __getitem__(self, key: str) -> io.BytesIO:
        if key not in self.files:
            raise KeyError()
        with ZipFile(self.path) as f:
            return io.BytesIO(f.read(key))
