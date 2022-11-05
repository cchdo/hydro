Basic CF/netCDF operators
=========================

This is a planning/ideas document.

Overview
--------
Manipulation of the CCHDO/ODF CF/netCDF format is needed to support data operations at sea and on shore.
On shore, CCHDO performs "data merges" where data submitted by program participants is integrated into single data files.
At sea, ODF is creating the initial data files and integrating onboard data similar to CCHDO.
Perhaps the largest difference is that ODF must create the basic profile records while CCHDO is often doing updates of an existing record.
To support both. a set of "low level" operations needs to be defined.

Here is a broad overview of what is needed:

* Initialize an "empty" dataset
* Add/Remove Profiles (N_PROF dim)
* Add/Remove vertical levels (N_LEVELS dim)
* Add/Remove Per Profile Vertical Levels (Z axis)
* Add/Remove non required variables
* Add/Remove ancillary variables
* Add/Remove variable data
* Add/Remove ancillary variable data

Initialize Empty Dataset
------------------------
A function initializing an empty dataset should return an xr.Dataset with the following properties:

* Contain 2 dimensions N_PROF and N_LEVELS with their values set to 0 (this might create a netCDF4 dataset with unlimited dims)
* Include all the required variables with the correct attrs, dims, and variable dtypes.
* Sets correct global attrs (TBD)

Add/Remove Profiles
-------------------
Adding a profiles requires that certain attributes about it are known before it can be created. These include:

* Expocode
* station
* cast
* longitude (X)
* latitude (Y)
* time (T)

The actual vertical level (Z), in our case pressure, is not needed at profile initialization time.
A function adding a profile should require the above coordinates and append the profile information to the end.
Optionally, it might sort the profiles by time.
All expanded arrays should have the new "slots" filled with an appropriate fill values, nan for numeric (even flags internally), and empty string for char arrays.

Removal of a profile should remove whatever it needs such that the profile is gone.
Optionally guard against deletion of non coordinate data

Add/Remove Vertical Levels
--------------------------
Due to the use of the incomplete multidimensional array representation of profiles (CF 9.3.2), it is valid for the Z coordinate to contain missing values as long as every other variable is missing the same data.
A function that adds vertical levels therefore is one that just expands the N_LEVELS dimension and adds the appropriate fill values in the new slots.
Example, it would make sense for a cruise is using a 36 place rosette to expand the N_LEVELS from 0 to 36 and not need to add any additional vertical levels for the remainder of the cruise, only adding profiles.

Removal of one or more vertical levels should ideally only be needed at the "end" of the array/profile.
Optionally guard against deletion of non coordinate data.

Add/Remove Per Profile Vertical Levels
--------------------------------------
The above Add/Remove Vertical Levels only creates the space in the data structures to hold the actual vertical axis data.
The use of an incomplete multidimensional array means not every profile will have the same number of vertical levels.
In the CF/netCDf format, a vertical level for a profile is considered available if and only if it has a value for "sample", this is true for CTD files as well as bottle.
Additionally, the vertical coordinate, pressure, must not have any fill values where there is also a "sample".

This means both "sample" and "pressure" are needed to create a valid vertical level "slot" in a profile.
The block of data needs to be contiguous, i.e. it starts from the 0 position in the array and ends at the n-1 index, where n is the actual number of vertical levels of the specific profile.
The Z values also need to be sorted from shallow to deep

Removal of a vertical level should probably be done by "sample".
If the last vertical level is not the one being removed, the resulting array needs to be rearranged so the data are contiguous.
The array shape would not change.
The removal of a vertical level would need to occur in all variables that are referenced.
Optionally can guard against deletion of non coordinate data

Possible Idea:
``````````````
Since two bits of information are needed, and their data types are known, perhaps the API might be one that accepts a python dict:

.. code::

    levels = {"1": 5000, "2", 4600.3}
    add_profile_level(level)

The add profile level function could also be the place the data are sorted.

Add/Remove non required variables
---------------------------------
The non required variables are what most people would consider to be the actual data in the file.
Things like temperature, salinity, oxygen, etc...
Adding a variable is one of the most basic operations in netCDF (there is a createVariable function) and for our purposes, involves setting the correct dtype, referencing the correct dims, and getting the proper attributes set.
The correct attributes depend on what the specific variable is.
These should reference the cchdo.params database until we have a well defined way of dealing with "non canonical" variables.

Removing a variable need to cleanup any ancillary variables that exclusively reference the removed variable.
Some optical parameters require cleanup of additional coordinate dimensions.

Add/Remove ancillary variables
------------------------------
Ancillary variables include flags, uncertainties, and in the case of many carbon parameters, the analytical temperature.
They are created/removed the same way as the variables above, however, the "parent variable" must already exist and be updated to reference the newly created ancillary variable.

Removal of an ancillary variable must cleanup any references to that ancillary variable.
There is not a one to one relation between variable and ancillary variables, e.g. a single flag variable might be referenced by multiple other variables.

Add/Remove variable data
------------------------
Adding and removing data is done using the (expocode, station, cast, sample) composite keys to reference specific cells and change their values.
Some variables need more coordinate information (e.g. wavelength) to get the specific cell.

Removal of variable data is done by setting the cell value to the appropriate fill values (nan or empty string) depending on variable dtype.

Optionally (perhaps by default), data changes should only be allowed where the flag ancillary variable suggests there should be values.

Variable data updates are closely tied with ancillary data updates, especially flags.
We probably want this function and the next one to actually be the same function.

Add/Remove ancillary variable data
----------------------------------
Ancillary variable data is indexed similarly to the variable data.
It is listed separately here because one of the earliest data operations that occurs is setting the flags where data are expected in the future.
ODF calls this "sample log entry".
The flag value indicates what variables collected water for analysis and is updated when the data actually arrive.
Flag updates also happen when QC is performed.

There is a situation where a problem was identified with the sampling device itself (niskin) and all water samples that came from that bottle should at least be flagged as "not good".
This has not been without disagreement, since the flags for variables are supposed to be about the specific measurement and not if that measurement was done on water that makes sense.
However, checking the "bottle flag" is a nuance missed on many users of the data.