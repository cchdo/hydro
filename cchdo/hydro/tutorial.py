import os
import io
import requests
from zipfile import ZipFile
from collections.abc import Mapping

from . import _hydro_appdirs

bottle_uri = "https://cchdo.ucsd.edu/search?q=a&download=exchange%2cbottle"
bottle_fname = "bottle_data.zip"


def _cache_dir():
    path = _hydro_appdirs.user_cache_dir
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


class CCHDOBottleData(Mapping):
    def __init__(self):
        self.path = os.path.join(_hydro_appdirs.user_cache_dir, bottle_fname)
        try:
            with ZipFile(self.path) as f:
                self.files = f.namelist()
        except FileNotFoundError:
            load_cchdo_bottle_data()
            with ZipFile(self.path) as f:
                self.files = f.namelist()

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for key in self.files:
            yield key

    def __getitem__(self, key):
        if key not in self.files:
            raise KeyError()
        with ZipFile(self.path) as f:
            return io.BytesIO(f.read(key))
