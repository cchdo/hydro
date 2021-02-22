import logging
from dataclasses import dataclass
from typing import Optional
from operator import methodcaller

import xarray as xr

# TODO Remove me
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
# end Remove

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckResult:
    error: Optional[str] = None
    warning: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.warning is None


class CCHDOnetCDF10:

    __cchdo_version__ = "1.0"

    def check_cf_version(self, ds):
        cf_version = "CF-1.8"
        try:
            conventions = ds.attrs["Conventions"]
        except AttributeError:
            return CheckResult(
                error=f"No 'attrs' attribute on {ds}, is it an xarray.Dataset?"
            )
        except KeyError:
            return CheckResult(error="Global attribute 'Conventions' is required")

        # We need to check that the correct cf conventions version is in the
        # "Conventions" attribute, since there can be more than one whitespace
        # seperated convention, we also need to check the split apart version
        # for exactly the value we expect
        if cf_version in conventions and cf_version in conventions.split():
            return CheckResult()

        return CheckResult(error="{cf_version} not in {conventions}")

    def iter_errors(self, ds: xr.Dataset):
        checks = {
            name: methodcaller(name, ds)
            for name in dir(self)
            if name.startswith("check_")
        }
        log.debug(f"Check methods: {list(checks.keys())}")
        for name, check in checks.items():
            check_result = check(self)
            log.debug(f"{name}: {check_result}")
            if check_result.ok:
                continue
            yield check_result

    def validate(self, ds: xr.Dataset):
        return not any(self.iter_errors(ds))


# Alias the "curren version"
CCHDOnetCDF = CCHDOnetCDF10
