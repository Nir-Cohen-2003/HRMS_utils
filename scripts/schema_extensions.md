# mzML Schema Extensions Guide

This document explains how to add additional fields when parsing mzML files in the HRMS_utils project.

## Overview

mzML files use a controlled vocabulary (CV) system from the PSI-MS ontology to represent metadata. Each piece of data is identified by an "accession" number (e.g., `MS:1000511` for ms level). The parser in `mzml_to_polars.py` extracts data by looking up these accession numbers.

## Architecture

### Key Files

- `revised_mzml_schema.xsd` - XSD schema documenting CV accessions and structure
- `mzml_to_polars.py` - Python parser using lxml

### Data Types in mzML

1. **cvParam** - Controlled vocabulary parameters with standardized accession numbers
2. **userParam** - Vendor-specific or custom parameters

## Adding New CV Parameters

### Step 1: Find the CV Accession

Look up the CV accession in the PSI-MS ontology:
- Browse: https://www.ebi.ac.uk/ols/ontologies/ms
- Search for your parameter (e.g., "ion mobility")

Example: Ion mobility drift time has accession `MS:1002476`.

### Step 2: Update the CV_ACCESSIONS Dictionary

In `mzml_to_polars.py`, add the new accession to the `CV_ACCESSIONS` dictionary:

```python
CV_ACCESSIONS = {
    # ... existing entries ...
    
    # Ion mobility (new)
    "ion_mobility_drift_time": "MS:1002476",
}
```

### Step 3: Add Extraction Logic

Add the extraction code in the appropriate section:

```python
# For spectrum-level parameters (inside the main loop)
ion_mobility = get_cv_value(spectrum, CV_ACCESSIONS["ion_mobility_drift_time"], float)
```

For parameters within specific elements like `precursor` or `activation`, search within those elements:

```python
# For precursor-level parameters
if precursors:
    precursor = precursors[0]
    ion_mobility = get_cv_value(precursor, CV_ACCESSIONS["ion_mobility_drift_time"], float)
```

### Step 4: Add to Output Dictionary

Add the field to `spec_data`:

```python
spec_data = {
    # ... existing fields ...
    "ion_mobility": ion_mobility,
}
```

### Step 5: Update Schema

Add the field to the Polars schema:

```python
schema = {
    # ... existing fields ...
    "ion_mobility": pl.Float64,  # or appropriate type
}
```

### Step 6: Document in revised_mzml_schema.xsd

Add the accession to the documentation section in the XSD file:

```xml
<!-- Ion Mobility CVParams -->
  MS:1002476 - ion mobility drift time
```

## Adding New userParam Fields

For vendor-specific parameters stored as `userParam`:

### Step 1: Identify the Parameter Name

Examine sample mzML files to find the exact `name` attribute:

```xml
<userParam name="[Thermo Trailer Extra]Monoisotopic M/Z:" value="378.20250389287577" type="xsd:float"/>
```

### Step 2: Add Extraction Logic

Use `get_user_param_value`:

```python
thermo_monoisotopic_mz = get_user_param_value(
    spectrum,
    "[Thermo Trailer Extra]Monoisotopic M/Z:",
    float
)
```

### Step 3: Follow Steps 4-5 Above

Add to output dictionary and schema.

## Common CV Accession Categories

### Spectrum Level
| Accession | Name | Type |
|-----------|------|------|
| MS:1000511 | ms level | int |
| MS:1000127 | centroid spectrum | bool (presence) |
| MS:1000128 | profile spectrum | bool (presence) |
| MS:1000129 | negative scan | bool (presence) |
| MS:1000130 | positive scan | bool (presence) |
| MS:1000285 | total ion current | float |
| MS:1000504 | base peak m/z | float |
| MS:1000505 | base peak intensity | float |

### Scan Level
| Accession | Name | Type |
|-----------|------|------|
| MS:1000016 | scan start time | float |
| MS:1000512 | filter string | str |
| MS:1000927 | ion injection time | float |

### Isolation Window
| Accession | Name | Type |
|-----------|------|------|
| MS:1000827 | isolation window target m/z | float |
| MS:1000828 | isolation window lower offset | float |
| MS:1000829 | isolation window upper offset | float |
| MS:1000794 | isolation window lower limit | float |
| MS:1000795 | isolation window upper limit | float |

### Selected Ion
| Accession | Name | Type |
|-----------|------|------|
| MS:1000744 | selected ion m/z | float |
| MS:1000041 | charge state | int |
| MS:1000042 | peak intensity | float |

### Activation/Fragmentation
| Accession | Name | Type |
|-----------|------|------|
| MS:1000045 | collision energy | float |
| MS:1000133 | CID | bool (presence) |
| MS:1000422 | HCD | bool (presence) |
| MS:1000598 | ETD | bool (presence) |

### Binary Arrays
| Accession | Name | Description |
|-----------|------|-------------|
| MS:1000514 | m/z array | Array contains m/z values |
| MS:1000515 | intensity array | Array contains intensity values |
| MS:1000521 | 32-bit float | Data encoded as float32 |
| MS:1000523 | 64-bit float | Data encoded as float64 |
| MS:1000574 | zlib compression | Data is zlib compressed |
| MS:1000576 | no compression | Data is not compressed |

## Example: Adding Total Ion Current

```python
# Step 2: Add to CV_ACCESSIONS
CV_ACCESSIONS = {
    # ... existing ...
    "total_ion_current": "MS:1000285",
}

# Step 3: Extract (spectrum level)
total_ion_current = get_cv_value(spectrum, CV_ACCESSIONS["total_ion_current"], float)

# Step 4: Add to spec_data
spec_data = {
    # ... existing ...
    "total_ion_current": total_ion_current,
}

# Step 5: Add to schema
schema = {
    # ... existing ...
    "total_ion_current": pl.Float64,
}
```

## Testing New Fields

After adding a new field:

1. Run the parser on a test file containing the new data:
   ```python
   df = mzml_to_polars_lxml("test_file.mzML")
   print(df.select(["id", "your_new_field"]).head())
   ```

2. Verify non-null values exist where expected:
   ```python
   print(df["your_new_field"].null_count())
   ```

3. Check value ranges are reasonable:
   ```python
   print(df["your_new_field"].describe())
   ```

## Handling Optional vs Required Fields

- Most CV parameters are optional in mzML
- Initialize fields as `None` before extraction
- Use nullable Polars types in schema
- The parser gracefully handles missing parameters

## XPath Tips for Complex Extraction

For nested elements, be specific about the search path:

```python
# Search anywhere in spectrum (less precise)
matches = spectrum.xpath(".//*[local-name()='cvParam' and @accession='MS:1000045']")

# Search only in activation element (more precise)
activations = precursor.xpath("./*[local-name()='activation']")
if activations:
    matches = activations[0].xpath(".//*[local-name()='cvParam' and @accession='MS:1000045']")
```

## Resources

- PSI-MS Controlled Vocabulary: https://www.ebi.ac.uk/ols/ontologies/ms
- mzML Specification: https://www.psidev.info/mzML
- Unit Ontology (for units): https://www.ebi.ac.uk/ols/ontologies/uo
