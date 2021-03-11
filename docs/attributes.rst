**********
Attributes
**********


Variable Level Attributes
=========================

``_FillValue``
--------------

:dtype:      same as the variable
:usage:      variables
:required:   only if there are missing data
:reference:  CF-1.8, NUG

CF Definiton
````````````
A value used to represent missing or undefined data. Allowed for auxiliary coordinate variables but not allowed for coordinate variables.

CCHDO Usage
```````````
Largely the same as before. For floating point type data (float,
double), the IEEE NaN value will be used.
