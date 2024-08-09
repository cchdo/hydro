==========================
Exchange Checker/Converter
==========================
The exchange checker/converter is a fully in browser (no server side processing) file converter for the WHP Exchange format to the newer CF/netCDF format.
It will also output the other legacy formats at CCHDO: WOCE, the COARDS netCDF formats, and a WOCE sum file.
This converter is only available in the html/browser versions of the documentation.

.. note::
    Processing a CTD file can take a long time and I don't yet know how to show progress in the browser.

.. warning::
    This has been pinned to an older version of cchdo.hydro while numpy 2 support is worked on in pyodide.

.. raw:: html
    :file: _exchange_checker_include.html 