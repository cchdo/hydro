# Metadata Model

:::{mermaid}

classDiagram
  namespace Classic {
    class DataVariable {
        attrs
    }
    class CoordinateVariable
    class QCVariable
    class ErrorVariable
    class AncillaryDataVariable
  }
  namespace Classic.Enhanced {
    class URLVariable {

    }
  }
  namespace Enhanced["Enhanced Metadata (uses netCDF groups)"] {
    class NamespaceVariable {
    
    }
    
    class MetadataGroup {

    }
    class ClassVariable {

    }
    class PropertyVariable {

    }
  }

  DataVariable "*" --> "0..1" MetadataGroup
  DataVariable "1" --> "0..1" ErrorVariable
  DataVariable "1" --> "0..1" AncillaryDataVariable
  DataVariable "1" --> "1" CoordinateVariable
  DataVariable "1" --> "0..1" QCVariable
  PropertyVariable "1" --> "1" ClassVariable
  ClassVariable "1" --> "*" PropertyVariable
  MetadataGroup "1" --> "*" ClassVariable
  PropertyVariable "1" --> "*" PropertyVariable
  DataVariable "1" --> "0..1" URLVariable
  MetadataGroup "1" --> "*" PropertyVariable
:::

In the above diagram, all the class boxes are netCDF variables.
The grouping of Classic represent the variable that are interchangeable with the legacy exchange tabular text format.
The Classic Enhanced grouping represents a structure that is being added and will have a representation inside the exchange files.
The Enhanced Metadata represents a generic structure of variables (tree or graph) that can encode any graph of metadata we would like to attach to one or more variables.
More explanation after some definitions.

## Definitions

### Data Variable
A data variable is the primary variable for some measurement and associated metadata.
It is a variable not referenced by any other variable as a CF Ancillary Variable (which includes the Error variable and QC variable in this model) or referenced as another variables CF coordinates.
This is the variable that contains "the data" such as silicate or nitrate values.
In some ways, a data variable is a variable that isn't something else rather than its own specific concept.
For the CCHDO purposes, the Data Variables are the entry point to the metadata graphs encoded with the netCDF files

### Error Variable
An error variable is a variable holding basic uncertainties (e.g. 1 stddev) for a Data Variable.
It MUST have the same units, and, as of this writing, have the same shape and sparseness of the data variable referencing it.
As of writing, CF is working to extend this to encode the various components of uncertainty.
This variable is referenced by a Data Variable via the data variable's ancillary_variables attribute.
It has a stable variable suffix within CCHDO netCDF files of `<primary>_error` where `<primary>` is the netCDF variable name of the data variable.

### Ancillary Data Variables
An ancillary data variable is a variable containing some other measured values associated with the primary data variable.
Examples include: pH temperature, equilibrator temperatures, oxygen draw temperatures, extra practical salinity measurements.
It "looks" like data, but should not be used outside of the context for which is an ancillary variable.
There is no current standard way of naming this variable, although the temperature version has a CF standard name, and as of this writing, a practical salinity name has been proposed.

### Coordinate Variable
In a CCHDO netCDF file, a coordinate variable is, to use the CF terminology, Auxiliary Coordinates.
They are any variable that appears in another variables `coordinates` attribute.
These provide temporal, spatial, or addressing information.
Examples includes: expocode, station, cast, sample, latitude, longitude, date, pressure.
These may also contain information about wavelengths or angles in the case of optical measurements.

### QC Variable
A QC variable is an ancillary variable that contains quality flag information.
It is currently required to have the same shape and sparseness as its referencing primary data variable.
The flag values and definitions are encoded using standard CF mechanisms

### URL Variable
A URL variable is a string ancillary variable that contains URLs, preferably DOIs, to the data that would normally appear in the netCDF file itself but cannot for some reason.
This variable MUST NOT be used for any other URLs than to data or data landing pages.
It is not meant to link to e.g. cruise reports, publications based on the data, etc., those should be encoded using the enhanced metadata groups and not the enhanced classic variable.
It is appearing in classic because it will be output with exchnage files once that support is added.

### Namespace Variable
The namespace variable in the enhanced metadata group is a single container variable with no data.
There MUST be one and only one namespace variable in the enhanced metadata group.
The attributes of this variable map attribute keys to url namespaces.
For example, "schema" might map to "https://schema.org", then in any other enhanced metadata variable, the attribute key "schema" could be replaced with "https://schema.org" and the attribute value put on the end of that URL to indicate a schema.org type or property.
E.g. "schema": "name" in netCDF would become "https://schema.org/name"

### Metadata Group Variable
The metadata group variable is the entrypoint for the enhanced metadata it references either Class or Property variables.
This variable allows multiple primary data variables to all reference a shared set of metadata.
For example, all the nutrient variables (silicate, nitrate, nitrite, phosphate) all usually have the same PI and have shared metadata fields.
All these primary data variables would reference the same group.
Data variables may reference more than one group if needed, for example if the programmatic information is the same (PI, award, etc..) but the processing level of the data varies by variable.

### Class Variable
A Class variable represents a collection of properties that make up some concept or type.
It should not have data by itself, but should instead reference property variables.
For example, https://schema.org/Person is a type/class.
Using the above Namespace example, a type of person SHOULD have a "schema" -> "Person" attribute mapping.

### Property Variable
A property variable represents a key of a class or type (represented by a class variable), it may either contain the data itself as data values or reference another class variable.
All leaf nodes must be property variables.

## Some Title Here (discussion?)
Given that the above structures could probably represent almost any metadata structure, the key to usefulness will be exactly defining some set of metadata we do want to represent without becoming too complicated.
We will want to do this incrementally as the requirements grow.
For CCHDO purposes, there should be only one way to represent something, and it should be consistent throughout the entire data holdings, not just a single dataset/file.

At this point, a question deserving answering is why not just stick and entire JSON-LD or other RDF serialization into the netCDF file.
The primary reason this cannot be easily done is that the granularity of the metadata properties are not fixed.
For example, the PIs of a collection of variables might change halfway though a dataset (per profile granularity).
Or the processing level (final/preliminary) may vary at the _data point_ level (per sample granularity).
We have additional requirements of needing to be able to split apart, or combine datasets.
Encoding the metadata in a way that allows us to take advantage of the netCDF structures, specifically xarray/numpy broadcasting rules allows these combinatorial problems to be addressed by the format itself.
