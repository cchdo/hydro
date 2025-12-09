# fq json
This document started as slides for a presentation on 2025-12-10

## "fq"?
fq is short for "[Fully Qualified](https://en.wikipedia.org/wiki/Fully_qualified_name)"

The idea being object completely identifies where the data value belongs.

## Some History
* Starts with the ODF Shipboard Database Written by Frank Delahoyde
  * Each Parameter had its own table in a postgres database
  * Shipboard data groups would submit parameter specific formats to the website
  * Raw sql files would be executed with these files as input (maybe some transform using tcl)
* In the ODF Bottle DB rewrite (kicked off on ARC01 2015), this parameter specific pattern was kept while the database backend was unified
  * A custom input format and processor needed to be defined on the server side
  * required server restarts, error handling (usually unhandled so 500 error), dealing with database calls, etc...

At some point during P06 (2017) a common intermediate format was developed so that scripts that translate parameter specific data inputs could be _external_ to the website code.
It was tied very closely to how the database was updated internally in the ODF bottle database, it was not intentionally designed for easy data input.
However, in practice the scripts that translated parameter specific formats ended up being very short and small.

Aside, why didn't we make some sort of prescribed format that the shipboard analysts must submit in?
The shipboard analysts are very busy people who are usually working with some custom and old acquisition.
They aren't usually programers and often have no ability to modify their software.
In some extreme cases, folks are writing down (with pens in a notebook) the data from a system that can only display data on a screen then typing this into another computer.
We aren't going to add to this stress in anyway and desire to make the data submission process as friendly as possible, even when not easy.

Over time, with the exception of the CTD/ODF parameters, all the custom input format scripts on the server side have been replaced with the fq json as the target.

The ODF Database is pure strings and has no concept of numeric values, as such, all the inputs values for parameters are expected to be string valued.
The ODF Database also does not store the expocode, so in the ODF versions of this, no expocode key exists.

:::::{grid} 2

::::{grid-item}

:::{code-block} json
:caption: Quoted string value
:emphasize-lines: 7
:lineno-start: 1

[
  {
    "EXPOCODE": "33RR20130321",
    "STNNBR": "1",
    "CASTNO": 1,
    "SAMPNO": "1",
    "OXYGEN [UMOL/KG]": "234.1",
    "OXYGEN [UMOL/KG]_FLAG_W": "2",
  }
]
:::

::::
::::{grid-item}

:::{code-block} json
:caption: Numeric literal
:emphasize-lines: 7
:lineno-start: 1

[
  {
    "EXPOCODE": "33RR20130321",
    "STNNBR": "1",
    "CASTNO": 1,
    "SAMPNO": "1",
    "OXYGEN [UMOL/KG]": 234.1,
    "OXYGEN [UMOL/KG]_FLAG_W": "2",
  }
]
:::

::::
:::::