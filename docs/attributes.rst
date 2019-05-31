Attributes
==========

``_FillValue``
--------------

:type:       numeric
:usage:      variables
:required:   only if there are missing data
:reference:  CF-1.8, NUG

CF Definiton
````````````
The smallest and the largest valid non-missing values occurring in the variable.

CCHDO Usage
```````````
Largely the same as before. For floating point type data (float,
double), the IEEE NaN value will be used.
