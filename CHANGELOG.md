# v1.0.2.13 (2025-06-28)
* Support specifying the text encoding of the input exchange file
* (Bug) Fix fill values in bottle date/time being printed as nan in generated exchange files
* Add the ability to ignore columns using the ODV parameter syntax to `read_exchange` and `read_csv`
* Add a --roundtrip flag to the status-exchange command, this will convert from the exchange file online at CCHDO to xarray/netCDF then back to exchange, then back to netCDF, the purpose is to check that the derived exchange is valid.
* Moved check_flags from exchange to checks

# v1.0.2.12 (2024-10-29)
* Add support for adding CDOM params/wavelengths
* Add support for merging CDOM in merge_fq
* (Bug) Fix merge_fq putting alternate paramter data in the wrong place
* (Bug) Fix a crash in the COARDS writer on some architectures (x64) when CTDNOBS has fill values
* Fix exception caused by string dtype parameters with all fill values

# v1.0.2.11 (2024-08-30)
* Removed 2 rouge debug print statemnets from digging into that WOCE bug

# v1.0.2.10 (2024-08-30)
* (Major Bug) Fix not calculating the correct string length for string fields that have an inconsistent length (e.g. station)
* (Bug) Fix the legacy woce writer not writing the data block if no quality flags are in the file

# v1.0.2.9 (2024-08-08)
* (New) Added :py:func:`hydro.core.add_param` and :py:func:`hydro.core.remove_param` functions
* (Bug) Fix crash in the COARDS writer when the comments are just an empty string
* Add cchdo.auth to cli optional requiremnets
* Vectorize the merge_fq accessor for greater speed
* use absolute imports throughout the library
* Speedups in string processing (precision extraction) from numpy 2

# v1.0.2.8 (2024-03-22)
* netCDF4 is now requried as part of the selftest option when installing

# v1.0.2.7 (2024-03-22)
* (Bug) fix to_exchange accessor failing for variables with seconds and the unit
* (Bug) fix to_coards accessor failing for variables with seconds and the unit
* Add status-cf-derived command that tests all all public CF files at CCHDO going from netCDF to every other supported format

# v1.0.2.6 (2024-03-18)
* Support for duplicate parameters
* (Bug) fix to_exchange accessor failing with a Dataset containing CDOM variables
* (Bug) fix for the flag column getting lost when alternate units for the same parameter were present in one file
  If, for example, a file had CTDTMP [ITS-90] and CTDTMP [IPTS-68] and both had CTDTMP_FLAG_W columns, only one of the parameters would get a flag column
* Added "coards" and "woce" file name generation support to `gen_fname()` accessor
* `to_woce()` now always returns zipfile bytes for ctd data
* Omit the "STAMP" text from generated WOCE files
* (changed) Bump min `cchdo.params` version to 2024.3

# v1.0.2.5 (2023-10-05)
* Rewrite the COARDS netCDF output to create xarray objects rather than netCDF datasets directly.
  In some quick testing, this results in about a 3x speed up, this depends more on variable count vs data length, so most of the performance increase is actually in the bottle output
  * Fixed a bug in COARDS where the fill value was not being set in the bottom depth variable
* Add `fill_values` and `precision_source` arguments to `read_csv`
* Add string literal types for the `ftype` parameter of `read_csv`
* CLI improvements:

  * made "precision_source" and option rather than positional argument
  * added a `--comments` option to allow the override of comments from either a string or file path prefixed with @.
  * Add a convert_csv subcommand which takes an additional ftype option to specify (C)TD or (B)ottle

* Removed the `matlab` optional install extra, this previously had a single dependency of "scipy" in it.
  Scipy is used by xarray for netCDF3 output so this dependency has been moved to the `netcdf` optional install extra.

# v1.0.2.4 (2023-09-05)
* (improved) the read_csv method now handles ctd data better, specifically you do not need to include a SAMPNO column if the FileType is CTD.
* Switched linting in pre-commit and CI to use ruff
* (changed) Bump min `cchdo.params` version to 2023.9

# v1.0.2.3 (2023-07-24)
* Add `read_csv` method
* (bug) Remove the `C_format` and `C_format_source` attributes for non floating point variables. Integer and string values are exact so do not need any sort of format hint. Including a format string for non floating point values is undefined behavior in the netCDF-C Library and can result in crashing.
* (new) Add `to_coards()` and `to_woce()` accessors to maintain legacy formats at CCHDO.
* (new) All the `to_*` accessors now support a path argument that will accept a writeable binary mode file like object or a filesystem path to write to.
* (new) Add a `compact_profile()` accessor that drops the trailing fill values from a profile
* (new) Add the a `file_seperator` and `keep_seperator` to `cchdo.hydro.exchange.read_exchange()`.
  The `keep_seperator` argument defaults to True.
  This is specifically to allow the reading of CTD exchange files that have been concatenated together (rather than zipped).
  Assuming there is nothing after "END_DATA" and you cat a bunch of _ct1.csv files together, they should be readable if "END_DATA" is passed into the `file_seperator` argument.
* (new) Add `--dump-data-counts` option to the exchange status generator which will dump a json document containing a object with nc_var name strings to count integers of how many 
  variables with this name actually contain any data (i.e. are not just entirely fill value).
* Add a `--version` option to the cli interface
* (changed) Export `read_exchange` from the top level `cchdo.hydro` namespace.
* (changed) Bump min `cchdo.params` version to 0.1.21
* (changed) Dropped netCDF4 as required for installation, if netCDF4 isn't installed already you can install with the `cchdo.hydro[netcdf4]` optional.

  * While this might seem like an odd choice for a library that started as one to convert WHP Exchange files to netCDf, netCDF 
    itself is not called until the very end of the conversion process. Internally, everything is an `xarray.Dataset`. This means you can
    install this library to read exchange files in tricky environments like pyodide or jupyterlite which already tend to have pandas and numpy in them.

* (bug) fix `pressure` variable not having a `_FillValue` attribute

# v1.0.2.2 (2022-08-18)
* Support for time values that are equal to 2400, when this is encountered, the date will be set to midnight of the next day.
* `read_exchange()` will now accept bytes and bytearray objects as input, wrapping data in an `io.BytesIO` is not needed anymore.

# v1.0.2.1 (2022-07-08)
* (breaking) fix misspelling of ``convert_exchange`` subcommand
* Will not rely on the python universal newlines for reading exchange data
* Will now combine CDOM parameters into a single variable with a new wavelength dimension in the last axis.
* Update the WHP error name lookup to be compatible with cchdo.params v0.1.18, this is now the minimum version
* Add an ``error_data`` attribute to ``ExchangeParameterUndefError`` that will contain a list of all the unknown ``(param, unit)`` pairs in an exchange file when attempting to read one.
* Add an ``error_data`` attribute to ``ExchangeDataFlagPairError`` that will contain a list of all the found flag errors as an xarray.Dataset
* Automatically attempt to use BTLNBR as a fallback if SAMPNO is not present in a bottle file.
* Automatically reconstruct the date of a missing BTL_DATE param if only BTL_TIME is present.
* Add ``--dump-unknown-params`` option to the status_exchange subcommand which will dump an unknown param list into a json format into the ``out_dir``.
* Performing a flag check is now behind a feature switch (defaults to true, for the status-exchange it is set to false)
* If a TIME column contains entirely the string "0" (not 0000) it will be ignored

# v1.0.2.0 (2022-04-12)
This release includes an almost complete rewrite of how the exchange to netCDF conversion works.
It now more directly uses numpy and has significant memory reduction and speed improvements when converting CTD (bottle is about the same).

* (breaking) The CLI was changed to support multiple actions which caused the exchange to netCDF functions to be moved to a sub-command "convert-exchnage" with the same interface as before.
* (breaking) The "source_C_format" attribute has been removed in favor of only having one "C_format" attribute, the "source" of the value in the C_format attribute will be listed in a new attribute "C_format_source" with the value of either "input_file" if the C_format was calculated from a text based input, or "database" if the C_format was taken from the internal database.
* (temporary) the netCDF to exchange function is not quite ready yet to work as an xarray accessor.
* (provisional) the order which netCDF variables appear is now in "exchange preferred" order.

## Bug Fixes
* Fixed an issue where the WOCE sumfile accessor would misalign latitude columns near the equator since they lacked a digit in the tens place.
* Fixed an issue where the WOCE sumfile accessor would use "pressure levels" of CTD source netCDF files as the number of bottles.
* Fixed an issue where stations might occur in an unexpected order.

# v1.0.1.3 (2021-08-25)
This release fixes many of the issues identified after the initial "1.0.0.0" release. Highlights include:

* Explicitly set the ``_FillValue`` attribute for the bottle closure time variable.
* The dtype for real number variables has been changed from ``float`` to ``double``
* If the source data is an "exchange csv", a ``source_C_format`` attribute will (with some exceptions) be present on the real number data variables.

# v1.0.1.2 (2021-03-11)
This release fixes a typo in the pyproject.toml file which would cause the _version.py file to be invalid.

# v1.0.1.0 (2021-03-11)
Hopefully this fixes the errors which prevented the project from being published automatically to pypi.

# v1.0.0.0 (2021-03-11)
After a whole bunch of testing, meetings, more testing, arguments, and a lot of work. We have declared the current status of the project as "good enough" for a 1.0.0 release.

There is much work to be done, especially since not all our files convert currently, but we think the ones that do convert are ready for public consumption. Unless something crazy goes wrong or is discovered, format changes should only be additive in nature (e.g. new attributes on variables).

The version will hopefully use the following (close to semver):

x.y.z

Where:

* x is incremented when a real breaking change to the netCDF output format is made.
* y is incremented when things are added to the netCDF format that should not break code which relies on previously existing attributes
* z is incremented for normal software releases that don't change the netCDF output.

:::{note}
The version number was since updated to be w.x.y.z where w.x is the CCHDO netCDF format version and y.z is the software versions
:::
