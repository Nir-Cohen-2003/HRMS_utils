# Formats API

Utilities for reading and processing various mass spectrometry data formats.

## MSDIAL

Functions for reading and processing MSDIAL chromatogram exports.

::: hrms_utils.formats.msdial.get_chromatogram
    options:
      show_root_heading: false

::: hrms_utils.formats.msdial.subtract_blank_frame
    options:
      show_root_heading: false

::: hrms_utils.formats.msdial.annotate_chromatogram_with_formulas
    options:
      show_root_heading: false

::: hrms_utils.formats.msdial.blank_config
    options:
      show_root_heading: false

## MSP/MSPEC (NIST)

Functions for reading NIST-format spectral libraries.

::: hrms_utils.formats.spectral_library.process_single_file
    options:
      show_root_heading: false

::: hrms_utils.formats.spectral_library.process_spectral_library
    options:
      show_root_heading: false

## MGF

Functions for reading Mascot Generic Format files.

::: hrms_utils.formats.mgf.read_mgf_to_dataframe
    options:
      show_root_heading: false

::: hrms_utils.formats.mgf.read_all_ms2_files
    options:
      show_root_heading: false

## See Also

- [Tutorial: MSDIAL Chromatogram Annotation](../tutorials/01-msdial-chromatogram-annotation.md)
- [Tutorial: MSP Library Processing](../tutorials/02-msp-library-processing.md)
- [Data Structures](../data-structures.md)
