GO-SHIP CF/netCDF Data Format Specification
===========================================

Introduction
------------
The traditional way that formats are thought about and described is via "file formats".
How the bytes arranged on disk, what the data types are, maybe even just some text with numeric characters in it.
Instead of focusing on how the data exist "at rest" or "on disk", netCDF instead describes a data model and an API (application programer interface) for data access.
Rather than specify the "on disk" format, we instead will specify a data model, and any format that supports the netCDF enhanced data model and can be accessed via the netCDF API is acceptable.
At the time of writing, the two most common at rest formats include HDF5, the traditional netCDF4 file, and zarr, a "cloud native" format designed for non disk/seekable storage mediums.

.. admonition:: Requirements

    The "on disk" or storage format is anything that supports:
    
    * Access via the netCDF4 software library API
    * The data and metadata model presented in this document


Requirement Levels
------------------
When specifying our requirements, we will follow the quitelines set in `BCP 14`_:

    The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in `BCP 14`_ [`RFC2119`_] [`RFC8174`_] when, and only when, they appear in all capitals, as shown here.

Additionally, we will try very hard to make sure all the requirements are in clearly marked admonitions.

.. danger::

    These requirement levels specify our "guarantees" when accessing the data and are specific to each published version.
    We will try very hard not to change things in a non backwards compatible way, but there may always be mistakes, bugs, community standards changes, etc., that require something breaking.

    If something is not specified here, you MUST assume it is undefined behavior that MAY change at any time.

Conventions
-----------
To increase data reusability and ease of access it is very useful to follow one or more community conventions when making decisions about data layout, what metadata to include, etc.
The  `CF Metadata Conventions`_ were chosen as the base for our data model, with influences of Argo and OceanSITES.
Specifically, we are using the `NetCDF Climate and Forecast (CF) Metadata Conventions version 1.8`_

.. admonition:: Requirements

    The data and metadata MUST conform to `NetCDF Climate and Forecast (CF) Metadata Conventions version 1.8`_

.. note::

    Internally, CCHDO is using `xarray`_ as the base for almost all the internal software working with netCDF.
    The internal data model of `xarray`_ is very close to but not exactly CF 1.8, and is a subset of what CF 1.8 allows.
    We have found that accepting the minor limitations of xarray to be worth the access to the large `scientific software ecosystem`_ that has developed around it.

    .. _xarray: https://docs.xarray.dev/en/stable/
    .. _scientific software ecosystem: https://docs.xarray.dev/en/stable/ecosystem.html

File Naming Conventions
-----------------------
When data are being distributed or shared using files, computer systems often rely on a file extension to identify the file type.

.. admonition:: Requirements

    As per `CF-1.8 Section 2.1`_, netCDF files SHOULD have the extension ``.nc``

At CCHDO, out usual data management granularity is cruise/leg, separated by discrete sample (bottle) and continuous sample (CTD) data types.
As a convenience, an additional suffix may be added to easily identify data containing only bottle or CTD data.

.. admonition:: Requirements

    * netCDF files containing exclusively bottle data MAY end with ``_bottle.nc``.
    * netCDF files containing exclusively ctd data MAY end with ``_ctd.nc``.
    * netCDF files containing mixed ctd and bottle data MUST NOT end with either ``_bottle.nc`` or ``_ctd.nc``.

Definitions
-----------
The terminology used to describe netCDF data tends to be very technical, with very specific definitions.
To confuse things, the netCDF user guide, the CF conventions, and popular software tools such as xarray all have slight variations on their usage of these definitions.
Due to this, we feel the need to include some of these definitions here.

dimension
  The netCDf data model is primarily concerned with storing data inside arrays, almost always this is numeric data.
  A netCDf dimension is the size of one side of these arrays and is given a name to reference it by.
  For example, a 2-d array of shape NxM has dimensions N and M.
  netCDf supports arrays with no dimensions, a scalar.

variable
  In a netCDF file, a variable is the most basic data object
  Variables have a name, a data type, a shape, some attributes, and the data itself.
  Variable names can be almost anything, the only character not allowed in a netCDF variable name is the forward slash "/".
  Names may start with or contain anything in unicode, they may not be valid variable names in your programing environment of choice.

  .. warning::

    It is also important to understand that variable names are simple labels and not data descriptors.
    If the name does have some human readable meaning, it often meant to help quickly identify which variables might be of interest, not describe the variable with scientific rigor.
    Do not rely on the inferred meaning of a variable name unless you have no other source of information (attributes, documentation, emails from colleagues, etc.).
  

.. _BCP 14: https://www.rfc-editor.org/info/bcp14
.. _RFC2119: https://datatracker.ietf.org/doc/html/rfc2119
.. _RFC8174: https://datatracker.ietf.org/doc/html/rfc8174
.. _CF Metadata Conventions: https://cfconventions.org/
.. _NetCDF Climate and Forecast (CF) Metadata Conventions version 1.8: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html
.. _CF-1.8 Section 2.1: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_filename
