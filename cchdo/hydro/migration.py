"""Functions that hopefully can migrate from a past version of data to a future version."""

from abc import ABC, abstractmethod

import xarray as xr


class MigrationABC(ABC):
    version_from = "1.0.0.0"

    def can_migrate(self, ds: xr.Dataset) -> bool:
        return self.version_from in ds.attrs.get("software_version", "")

    @abstractmethod
    def migrate(self, ds: xr.Dataset) -> xr.Dataset:
        return ds
