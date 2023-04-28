.. _attributes:

**********
Attributes
**********


Variable Level Attributes
=========================

``_FillValue``
--------------

:dtype:      same as the variable
:required:   only if there are missing data
:reference:  CF-1.8, NUG

CF Definiton
````````````
A value used to represent missing or undefined data. Allowed for auxiliary coordinate variables but not allowed for coordinate variables.

CCHDO Usage
```````````
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
````````````
Not used or defined in the CF conventions.

CCHDO Usage
```````````
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
````````````
Not used or defined in the CF conventions.

CCHDO Usage
```````````
For this variable, the value which would appear in the units line of the WHP Exchange or WOCE sea/ctd file.
Forms a pair with :ref:`whp_name`
Usage is the same as ``whp_name``


``standard_name``
-----------------
:dtype:      char
:required:   conditionally (see CF Usage)
:reference:  CF 1.8

CF Definiton
````````````
.. todo::
  get cf definiton

CCHDO Usage
```````````
The CF usage will be followed, if a CF standard name exists for physical quantity represeted by a variable, the most specific name MUST be used and appear in the ``standard_name`` attribute.
The CF standard names table is updated frequently, as names are added they will be evaluated for including in the CCHDO netCDF files to both be more specific or to add a standard name to a variable that did not have one previously.
Always check the param version attribute to see which version of the standard name table is in use for a particular file.

It is important to understand that standard names represet the physical quantity of the variable and not how the data was made.
Standard names cannot distinguish between salinity measured in situ with a CTD, salinity measured with an autosal, or even salinity from a model output.
The names are meant to help with intercomparison of the values themselves, not methods of determing that value.


``units``
-----------------
:dtype:      char
:required:   conditionally
:reference:  CF 1.8

CF Definiton
````````````
.. todo::
  get cf definiton

CCHDO Usage
```````````
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
````````````
This attribute is not defined in CF. 

CCHDO Usage
```````````
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
````````````
``C_format`` is not mentioned at all in the CF-1.8 docs.

CCHDO Usage
```````````
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
````````````
This attribute is not used in CF.

CCHDO Usage
```````````
This attribute describes where the value in ``C_format`` came from.
This attribute will only have the values of either ``"database"`` to indicate the ``C_format`` was taken from the internal parameters table, or ``"source_file"`` if the values was calcualted from input text.

``geometry``
------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````

``_Encoding``
-------------
:dtype:      char
:required:   no
:reference:  ref?

CF Definiton
````````````
This is not defined by CF, it is however a reserved attribute in `Appendix B`_ of the netCDF4-C manual.

.. _Appendix B: https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html

CCHDO Usage
```````````
This attribute is set by the libraries we use to make our data.
It will almost always be set on string or char array data with a value of "utf8".

``coordinates``
---------------
:dtype:      char
:required:   conditionally
:reference:  CF 1.8

CF Definiton
````````````

CCHDO Usage
```````````

``ancillary_variables``
-----------------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````

``flag_values``
---------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````

``flag_meanings``
-----------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````

``conventions``
---------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````

``resolution (time)``
---------------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````

``axis``
--------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````

``calendar``
------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````

``positive``
------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````

``geometry_type``
------------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````

``node_coordinates``
--------------------
:dtype:      dtype
:required:   maybe
:reference:  Ref

CF Definiton
````````````

CCHDO Usage
```````````


.. todo::
  Attrs:

  Global Level:

  * Conventions
  * cchdo_software_version
  * cchdo_parameters_version
  * comments
  * featureType

  ACDD Things we want at variable level:

  * creator_name
  * creator_email
  * processing_level
  * instrument
  * instrument_vocabulrary
  * comments (more of them)
  * contributor_name
  * contributor_email
  * contributor_role

  Non ACDD thing var level:
  
  * program_group

  Non ACDD global level?:
  
  * platform (ICES ship code)
  * start/end ports
  * actual start/end dates

  Huge TODO... history at the var and global level, including seperation between metadata and data history.
