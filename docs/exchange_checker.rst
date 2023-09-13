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

    <link rel="stylesheet" href="https://pyscript.net/latest/pyscript.css" />
    <script defer src="https://pyscript.net/latest/pyscript.js"></script>

    <p>
        cchdo.hydro version: <span id="hydro_version"></span><br />
        cchdo.params version: <span id="params_version"></span>
    </p>
    <p>
    <label>Add an exchange file (csv or zip) <input type="file" id="ex_file" name="ex_file"></label>
    <h4>Options</h4>
    <label><input id="checks_flags" type="checkbox" checked> Check Flags</label>
    <p>
    <button class="sd-sphinx-override sd-btn sd-text-wrap sd-btn-primary reference internal" id="process_exchange" py-click="_process_exchange()">Process Exchange</button>
    </p>
    <p>
    <div id='output'>
        <span id="status"></span>
        <br />
    </div>
    <py-script>
        from js import document, console, window, Uint8Array, Blob
        from pyodide.ffi import create_proxy
        import asyncio
        import io
        import traceback

        from cchdo.hydro import read_exchange
        from cchdo.hydro import accessors
        
        from cchdo.hydro import __version__ as hydro_version
        from cchdo.params import __version__ as params_version

        Element("hydro_version").element.innerText = hydro_version
        Element("params_version").element.innerText = params_version

        import logging
        import sys

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)

        def to_xarray_callback(arg):
            bytes = bytearray(Uint8Array.new(arg))
            ex_bytes = io.BytesIO(bytes)
            status = Element("status")
            check_flags = Element("checks_flags").element.checked
            checks = {
              "flags": check_flags
            }
            try:
                ex = read_exchange(ex_bytes, checks=checks)
            except ValueError as er:
                traceback.print_exception(er)
                status.element.innerText = f"Failure see traceback..."
                Element("process_exchange").element.disabled = False
                return
            status.element.innerText = f"Success, generating files"
            try:
                ex.to_netcdf("out.nc", engine="h5netcdf")
                with open("out.nc", "rb") as f:
                    nc = f.read()
                nc_blob = Blob.new([Uint8Array.new(nc)], {type : 'application/netcdf'})
                nc_url = window.URL.createObjectURL(nc_blob) 
                nc_download_link = document.createElement("a")
                nc_download_link.href = nc_url
                nc_fname = ex.cchdo.gen_fname()
                nc_download_link.download = nc_fname
                nc_download_link.innerText = f"Download CF/netCDF: {nc_fname}"
                output = Element("output")
                output.element.appendChild(nc_download_link)
                output.element.appendChild(document.createElement("br"))
                 
            except Exception as er:
                status.element.innerText = f"Could not generate WOCE"
            try:
                woce = ex.cchdo.to_woce()
                woce_blob = Blob.new([Uint8Array.new(woce)], {type : 'application/octet-stream'})
                woce_url = window.URL.createObjectURL(woce_blob) 
                woce_download_link = document.createElement("a")
                woce_download_link.href = woce_url
                woce_download_link.download = "woce_output.txt"
                woce_download_link.innerText = "Download Woce (might be txt or zip)"
                output = Element("output")
                output.element.appendChild(woce_download_link)
                output.element.appendChild(document.createElement("br"))
            except:
                status.element.innerText = f"Could not generate WOCE"

            Element("process_exchange").element.disabled = False

        def _process_exchange():
            Element("process_exchange").element.disabled = True
            try:
                status = Element("status")
                status.element.innerText = "Processing..."
                file_list = Element("ex_file").element.files
                first_item = file_list.item(0)

                first_item.arrayBuffer().then(to_xarray_callback)
            except:
                status.element.innerText = "Error, was a file picked?"
                Element("process_exchange").element.disabled = False

    </py-script>
    <h4>Python Log Console</h4>
  <py-terminal true></py-terminal>
  <py-config type="toml">
    packages = ["xarray", "cchdo.hydro", "h5netcdf"]
  </py-config>