*******************************************
GO-SHIP CF/netCDF Data Format Specification
*******************************************

Introduction
============
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
==================
When specifying our requirements, we will follow the guidelines set in `BCP 14`_:

    The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in `BCP 14`_ [`RFC2119`_] [`RFC8174`_] when, and only when, they appear in all capitals, as shown here.

Additionally, we will try very hard to make sure all the requirements are in clearly marked admonitions.

.. danger::

    These requirement levels specify our "guarantees" when accessing the data and are specific to each published version.
    We will try very hard not to change things in a non backwards compatible way, but there may always be mistakes, bugs, community standards changes, etc., that require something breaking.

    If something is not specified here, you MUST assume it is undefined behavior that MAY change at any time.

Conventions
===========
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
=======================
When data are being distributed or shared using files, computer systems often rely on a file extension to identify the file type.

.. admonition:: Requirements

    As per `CF-1.8 Section 2.1`_, netCDF HDF5 files SHOULD have the extension ``.nc``

At CCHDO, out usual data management granularity is cruise/leg, separated by discrete sample (bottle) and continuous sample (CTD) data types.
As a convenience, an additional suffix may be added to easily identify data containing only bottle or CTD data.

.. admonition:: Requirements

    * netCDF files containing exclusively bottle data MAY end with ``_bottle.nc``.
    * netCDF files containing exclusively ctd data MAY end with ``_ctd.nc``.
    * netCDF files containing mixed ctd and bottle data MUST NOT end with either ``_bottle.nc`` or ``_ctd.nc``.

Definitions
===========
The terminology used to describe netCDF data tends to be very technical, with very specific definitions.
To confuse things, the netCDF user guide, the CF conventions, and popular software tools such as xarray all have slight variations on their usage of these definitions.
Due to this, we feel the need to include some of these definitions here.

.. glossary::

  dataset
    A dataset is the point of entry for the netCDF api.
    Datasets contain :term:`groups<group>`, :term:`dimensions<dimension>`, :term:`variables<variable>`, and :term:`attributes<attribute>`.

  group
    :term:`Dimensions<dimension>`, :term:`variables<variable>`, and :term:`attributes<attribute>` can all be organized in a hierarchical structure within netCDF.
    This is similar to how files and directories exist on a computer filesystem.
    All netCDF4 files have at least one root group called "/" (forward slash).
    Currently, no addtional groups are used in GO-SHIP netCDF files.
  
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
  
  ancillary variable
    In CF, an ancillary variable is still a normal variable described above, but it contains information about other variables.
    Perhaps the most common example of an ancillary variable is the quality control flag, but also include information such as uncertainties.
    Some of the carbon data have strong temperature dependencies and so the temperature of analysis might be reported along side in an ancillary variable.
    
  coordinate
    Coordinates are variables that provide the labels for some axis, usually for identifying data in space and time.
    The typical examples of coordinates are longitude (X-axis), latitude (Y-axis), and time (T-axis).
    The vertical coordinate is a little more varied, usually oceanographic observation data will use pressure as the Z-axis coordinate.
  
    Xarray calls these "coordinates"
  
  coordinate variables
    In many netCDF aware applications there is a special case of variables called "coordinate variables" or "Dimension coordinate".
    The technical way you will see this defined is as a single dimensional variable that has the same name as its dimension.
    There tend to be other rules most programs enforce: there must be no missing values, values must be numeric, and values must be monotonic.
    These are most useful when the data occur on some regular grid.
  
    Perhaps a good way to think of coordinates variables is as the values the ticks would be in a figure plot.
  
    Xarray calls these "Dimension coordinates" and will be shown with a little asterisk ``*`` when exploring an xarray Dataset.
  
  auxiliary coordinate
    Auxiliary coordinates or "Non-dimension coordinates" are variables that do not share the same names as a dimension.
    These variables still label axes, but are more flexible for when the data do not occur on a regular grid or when there are multiple sets of coordinates in use.
    Auxiliary coordinates may be multidimensional.
    CF requires auxiliary coordinates to appear in the ``coordinates`` attribute of the variables it labels.
  
    Xarray calls these "Non-dimension coordinates" and will not have an asterisk next to their names when exploring an xarray dataset.
  
  attribute
    Attributes are extra pieces of data that are attached to each variable and is where the flexibility of netCDF to describe data is greatly enhanced.
    Attributes may also be attached at the "global" level
    Attributes are simple "key" to "value" mappings, the computer science term for these is "associative array".
    Python and Julia calls these "dictionaries", in matlab these are usually "Structure Arrays".
  
    Most of the focus of the common community data standards, CF, ACDD, OceanSITES etc., are on defining attribute keys, values, and how to interpret them.
    CF defines and controls attributes important to CF, but then allows any number of extra attributes.

Dataset Structure
=================
.. todo:: 

    write overview

    * Global attributes
    * Required variables
    * Technical variables and attrs (the geometry ones)
    * Notes on strings and chars
      * Encoding, line endings
      * where are actual strings allowed, netCDF4 python forces string types if non ascii

The CF conventions document is long, verbose, and (we think) intimidating at first glance.
This is due to the wide range of data structures supported by CF, and the need to carefully describe things in detail.
It is hard to know what parts are important for your, or our, data.
For any given dataset, only a small portion of the CF conventions will be used.
This is true not just for GO-SHIP data, but any data claiming to be compatable with CF.
We selected what we hope will be an easy entry point into the data stored in this standardized structure.

Chapter 9 of the CF conventions define what are called discrete sampling geometries, often refered to as a DSG.
Specifically, we selected the incomplete multidimensional array representation defined in 9.3.2 (TODO Ref).
This representation has two primary dimmensions, one of the profile and the other as the vertical level in that profile.
When each profile has different number of vertical levels, fill values will be in the trailing data slots.

Dimensions
==========
There are two basic dimensions in the data file, how many profiles there are, and how many vertical levels there are.
The two dimension names match the dimenion names found in argo profile files: N_PROF and N_LEVELS.

While netCDF4 supports an actual string data type, for compatibility and compression reasons, character arrays will be used to represent text data.
Character arrays have the string length as their last dimension, the number and values of these string dimensions is currently uncontrolled (xarray sets these automatically).
All char arrays or strings will be UTF-8 encoded.


.. admonition:: Requirements

    * There MUST be a dimension named ``N_PROF`` that describes the first axis of variables with a "profile" dimension.
    * There MUST be a dimension named ``N_LEVELS`` that describes the first axis of variables with no "profile" dimension, or the second axis of variables with a "profile" dimension
    * There MAY be zero or more string length dimensions.
    * Extra dimensions MAY exist if needed by data variables, these extra names are not standardized.
    * Any char array or strings, both in variable and attributes, MUST be UTF-8 encoded and MUST NOT have a byte order mark.

.. note::

    There is currently a single variable which requires an additional dimension to describe the radiation wavelength of its measurement.
    This dimension is currently called ``CDOM_WAVELENGTHS`` and is stored as the only coordinate variable in use.
    The actual relationship between the parent variable and this coordinate is contained in attributes defined by the CF conventions.

Global Attributes
=================
Attributes are bits of metadata with a name and a value attached to it.
Almost all the "work" being done by the CF conventions and other metadata standards are happening in the attributes, CF for example, does not standardize the variable names at all.

Global attributes contain information that applies to the entire dataset.
Some of these are defined by community standards, other by this document for internal use.
The following, case sensitive, global attributes are REQUIRED to be present:

``Conventions``
  Conventions is a char array listing what community standards and their versions are being followed.
  It MUST have the value ``"CF-1.8 CCHDO-1.0"`` and will change as new conventions are adopted
``featureType``
  The feature type char array attribute comes from the CF conventions section about discrete sampling geometries.
  It MUST have the value ``"profile"``
``cchdo_software_version``
  The cchdo software version is a char array containing the version of the cchdo.hydro library used to create or manipulate the dataset.
  It currently takes the form of ``"hydro w.x.y.z"`` where w.x is the data conventions version, and y.z is the actual software library version.
``cchdo_parameters_version``
  The cchdo parameters version char array contains the version for the internal parameters database the software was using at the time of dataset creation or manipulation.
  It currently takes the form of ``"params x.y.z"``.

The following, case sensitive, global attributes are OPTIONAL:

``comments``
  Comments human readable string containing information not captured in any other attributes or variables.

.. admonition:: Requirements

    * There MUST be a ``Conventions`` global attribute char array with space separate convention version strings defined by those conventions.
    * There MUST be a ``featureType`` global attribute char array with the value "profile".
    * There MUST be a ``cchdo_software_version`` global attribute char array with the version string of the cchdo.hydro software.
    * There MUST be a ``cchdo_parameters_version`` global attribute char array with the version string of the cchdo.params database.
    * There MAY be a ``comments`` attribute with more information. This attribute MAY be a string rather than a char array if there are non ASCII code points present.

Variable Attributes
===================
.. todo:: 

    Attrs to talk about:

    * whp_name
    * whp_unit
    * geometry
    * _Encoding
    * coordinates
    * ancillary_variables
    * standard_name
    * flag_values
    * flag_meanings
    * conventions
    * resolution (time)
    * axis
    * units
    * calendar
    * C_format
    * C_format_source
    * positive
    * reference_scale
    * geometry_type
    * node_coordinates

Variable attributes are like the global attributes, but instead of being attached to the entire dataset, are attached to variables.
These attributes are where almost all the metadata about a variable exist, things such as what the units of the measuremnet are or what the flag values mean.

``_FillValue``
--------------

:dtype:      same as the variable
:required:   only if there are missing data
:reference:  CF-1.8, NUG

CF Definiton
^^^^^^^^^^^^
A value used to represent missing or undefined data. Allowed for auxiliary coordinate variables but not allowed for coordinate variables.

CCHDO Usage
^^^^^^^^^^^
For floating point type data (float, double), the IEEE NaN value will be used.
Woce flag variables will be initialized with the value `9b`.
Some special coordinate variables are not allowed to have any ``_FillValue`` values in them

The ``_FillValue`` attribute has special meaning to the netCDF4 libraries (C and Java).
When the size of the variable is known (i.e. the variable does not have an "unlimited" dimmension) at the time the netCDF file is written, all of the space in the variable will be initalized with the value in ``_FillValue``.
This is usually almost entirely transparent to you the user, some software will change the data type when a variable still contains ``_FillValue`` values.
Matlab for example, will change byte (integers between 0 and 255) data into IEEE floating point values while replacing the fill value with NaNs.

.. _whp_name:

``whp_name``
------------
:dtype:      char or array of strings
:required:   conditionally (see CCHDO Usage)
:reference:  CCHDO

CF Definiton
^^^^^^^^^^^^
Not used or defined in the CF conventions.

CCHDO Usage
^^^^^^^^^^^
This attribute contains the name this variable would have inside a WHP Exchange or WOCE sea/ctd file.
Forms a pair with :ref:`whp_unit`
This attribute will only be on variables which are data, and not on flag variables or certain specal variables meant to be interpreted by CF compliant readers (e.g. ``geometry_container``).

Some variables cannot be represented by a single column in the WHP Exchange format, when this occurs, the attribute will be an array of strings containing all the names needed to represent this variable in WHP Exchange format.
The most frequent example will be the ``time`` variable, in WHP Exchange files, this may either be a pair of columns (DATE, TIME) or a single column (DATE) when time of day is not reported.
This will very likly be used to represet ex and em wavelengths for optical sensors with multiple channels.

.. warning::

  There is no requiremnt that all variables in a netCDF file contain unique ``whp_name`` and ``whp_unit`` pairs.


.. _whp_unit:

``whp_unit``
------------
:dtype:      char or array of strings
:required:   conditionally (see CCHDO Usage)
:reference:  CCHDO

CF Definiton
^^^^^^^^^^^^
Not used or defined in the CF conventions.

CCHDO Usage
^^^^^^^^^^^
For this variable, the value which would appear in the units line of the WHP Exchange or WOCE sea/ctd file.
Forms a pair with :ref:`whp_name`
Usage is the same as ``whp_name``


``standard_name``
-----------------
:dtype:      char
:required:   conditionally (see CF Usage)
:reference:  CF 1.8

CF Definiton
^^^^^^^^^^^^
.. todo::
  get cf definiton

CCHDO Usage
^^^^^^^^^^^
The CF usage will be followed, if a CF standard name exists for physical quantity represeted by a variable, the most specific name MUST be used and appear in the ``standard_name`` attribute.
The CF standard names table is updated frequently, as names are added they will be evaluated for including in the CCHDO netCDF files to both be more specific or to add a standard name to a variable that did not have one previously.
Always check the param version attribute to see which version of the standard name table is in use for a particular file.

It is important to understand that standard names represet the physical quantity of the variable and not how the data was made.
Standard names cannot distinguish between salinity measured in situ with a CTD, salinity measured with an autosal, or even salinity from a model output.
The names are meant to help with intercomparison of the values themselves, not methods of determing that value.


``units``
---------
:dtype:      char
:required:   conditionally
:reference:  CF 1.8

CF Definiton
^^^^^^^^^^^^
.. todo::
  get cf definiton

CCHDO Usage
^^^^^^^^^^^
The units attribute will follow CF.
The value must be physically comparible with the canonical units of the ``standard_name``.
The value will be the ``whp_unit`` translated into SI.

Unitless parameters will have the symbol "1" as their units.

.. todo::
  get ref to SI paper

Some examples:

*  discintigrations per minute (DPM) will be translated to their equivalent Bq, which will be scaled (1DPM = 0.0166 Bq)
* Practical salinity will have the units of "1", not variabtions on "PSU" or even "0.001" implying g/kg of actual salinity.
* Tritium Units are really parts per 1e18, so the equivalent SI units are the recriprical: 1e-18


``reference_scale``
-------------------
:dtype:      char
:required:   conditionally
:reference:  OceanSITES 1.4

CF Definiton
^^^^^^^^^^^^
This attribute is not defined in CF. 

CCHDO Usage
^^^^^^^^^^^
.. todo::
  get OceanSITES definition.

Some variables (e.g. temperature) are not described well enough by their units and standard name alone.
For example, depending on when it was measured, the temperature sensors may have been calibrated on the ITS-90, IPTS-68, or WHAT_WAS_BEFORE_t68 calibration scales.
While all the temperatures are degree C, users doing precice work need to know the difference.

.. todo::
  this is a controlled list internally, list which variables have a scale and what their value can be.


``C_format``
------------
:dtype:      char
:required:   no
:reference:  NUG

CF Definiton
^^^^^^^^^^^^
``C_format`` is not mentioned at all in the CF-1.8 docs.

CCHDO Usage
^^^^^^^^^^^
The ``C_format`` attribute will contain the format string from either the internal database of parameters or calcualted when converting from a text input.
The presence or lack of presence of this attribute will not change the underlyying values in the variable (e.g. you cannot round the values to the nearst integer using C_format).
This attribute is sometimes used when _displaying_ data values to a user.
When performing calculations in most software, the underlying data values are almost always used directly.
The only software we have seen respect the ``C_format`` attribute is ncdump when dumping to CDL.

If the data soure for this variable came from a text source, the ``C_format`` will contain the format string which represents the largest string seen.
For example, if a data source had text values of "0.001" and "0.0010", the ``C_format`` attribute would be set to ``"%.4f"``.
This can be tricky for data managers: if for example, the data source was an excel file, it is important to use the underlying value as the actual data and not a copy/paste or text based export.


.. warning::
  Use ``C_format`` as implied uncertanty if you have `no other` source of uncertanty (including statistical methods across the dataset).

  Historically, storing numeric values in text and the cost of storage meant there was a tradeoff between cost and precision.
  When looking though our database of format strings, the text print precision was almost always set at one decimal place more than the actual measuremnt uncertanty.
  Having these values published in the WOCE manual also lead to values being reported a certain way to conform to the WOCE format, which disconnected "print precision" from uncertanty.
  Additionally, the WOCE format was designed when IEEE floating point numbers were quite new.

  More recent measuremnets have started to include explicit uncertanties which will be reported along side the data values.
  Often, the cruise report will contain some charicterizaion of the uncertanty of a given measumrnet.


``C_format_source``
-------------------
:dtype:      char
:required:   yes if C_format is present
:reference:  CCHDO

CF Definiton
^^^^^^^^^^^^
This attribute is not used in CF.

CCHDO Usage
^^^^^^^^^^^
This attribute describes where the value in ``C_format`` came from.
This attribute will only have the values of either ``"database"`` to indicate the ``C_format`` was taken from the internal parameters table, or ``"source_file"`` if the values was calcualted from input text.

----

``geometry``
------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``_Encoding``
-------------
:dtype:      char
:required:   no
:reference:  ref?

CF Definiton
^^^^^^^^^^^^
This is not defined by CF, it is however a reserved attribute in `Appendix B`_ of the netCDF4-C manual.

.. _Appendix B: https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html

CCHDO Usage
^^^^^^^^^^^
This attribute is set by the libraries we use to make our data.
It will almost always be set on string or char array data with a value of "utf8".

``coordinates``
---------------
:dtype:      char
:required:   conditionally
:reference:  CF 1.8

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``ancillary_variables``
-----------------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``flag_values``
---------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``flag_meanings``
-----------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``conventions``
---------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``resolution (time)``
---------------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``axis``
--------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``calendar``
------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``positive``
------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``geometry_type``
------------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

``node_coordinates``
--------------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
^^^^^^^^^^^^

CCHDO Usage
^^^^^^^^^^^

Required Variables
==================
The following variables are required in all files:

* ``geometry_container``
* ``profile_type``
* ``expocode``
* ``station``
* ``cast``
* ``sample``
* ``longitude``
* ``latitude``
* ``pressure``
* ``time``

.. _BCP 14: https://www.rfc-editor.org/info/bcp14
.. _RFC2119: https://datatracker.ietf.org/doc/html/rfc2119
.. _RFC8174: https://datatracker.ietf.org/doc/html/rfc8174
.. _CF Metadata Conventions: https://cfconventions.org/
.. _NetCDF Climate and Forecast (CF) Metadata Conventions version 1.8: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html
.. _CF-1.8 Section 2.1: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_filename
