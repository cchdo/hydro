==========================
Exchange Checker/Converter
==========================
The exchange checker/converter is a fully in browser (no server side processing) file converter for the WHP Exchange format to the newer CF/netCDF format.
It will also output the other legacy formats at CCHDO: WOCE, and hopefully soon, the COARDS netCDF formats.
This converter is only available in the html/browser versions of the documentation.

.. note::
    Processing a CTD file can take a long time and I don't yet know how to show progress in the browser.
    
    Right now pyodide (and therefore this page) does not support the actual netCDF4 library we would need to make COARDS netCDF files.

.. raw:: html
    :file: _exchange_checker_include.html 