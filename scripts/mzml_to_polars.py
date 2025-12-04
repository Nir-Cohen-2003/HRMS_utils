import marimo

__generated_with = "0.18.2"
app = marimo.App()


@app.cell
def _():
    import pymzml
    import polars as pl
    import pandas as pd
    import numpy as np
    import os
    import time
    import lxml.etree as etree
    import base64
    import zlib
    import struct
    return base64, etree, os, pl, pymzml, struct, time, zlib


@app.cell
def _(os, pl, pymzml):
    def mzml_to_polars(mzml_path):
        if not os.path.exists(mzml_path):
            raise FileNotFoundError(f"File not found: {mzml_path}")

        data = []
        run = pymzml.run.Reader(mzml_path)

        print(f"Reading {mzml_path}...")
        printed_attrs = False
        for spectrum in run:
            scan_time = None
            if spectrum.scan_time:
                scan_time = spectrum.scan_time[0]
            if not printed_attrs and spectrum.ms_level == 2:
                print("Spectrum attributes:")
                # Collect non-private attributes for deterministic output
                attributes = sorted(
                    [attr for attr in dir(spectrum) if not attr.startswith("_")]
                )
                # Why: Accessing some properties may raise exceptions or be expensive; wrap access in try/except
                for attr in attributes:
                    print(f" - {attr}:")
                    try:
                        value = getattr(spectrum, attr)
                        # Avoid printing extremely large values unboundedly (helps readability)
                        val_repr = repr(value)
                        max_chars = 1000
                        if len(val_repr) > max_chars:
                            val_repr = val_repr[:max_chars] + "... (truncated)"
                        print(f"   {val_repr}")
                    except Exception as e:
                        print(f"   Failed to access '{attr}': {e}")
                printed_attrs = True
                printed_attrs = True
            mz_array = spectrum.mz
            intensity_array = spectrum.i

            spec_data = {
                "id": str(spectrum.ID),
                "ms_level": spectrum.ms_level,
                "scan_time": scan_time,
                "mz": mz_array,
                "intensity": intensity_array,
                "precursor_mz": None,
                "isolation_window_lower_bound": None,
                "isolation_window_upper_bound": None,
            }

            data.append(spec_data)

        schema = {
            "id": pl.String,
            "ms_level": pl.Int64,
            "scan_time": pl.Float64,
            "mz": pl.List(pl.Float64),
            "intensity": pl.List(pl.Float64),
            "precursor_mz": pl.Float64,
            "isolation_window_lower_bound": pl.Float64,
            "isolation_window_upper_bound": pl.Float64,
        }

        df = pl.DataFrame(data, schema=schema, orient="row")
        return df
    return


@app.cell
def _(base64, etree, os, pl, struct, zlib):
    def mzml_to_polars_lxml(mzml_path: str) -> pl.DataFrame:
        """
        Parse an mzML file using lxml and return a Polars DataFrame.

        Extracts spectrum metadata, precursor information, and binary data arrays.
        Uses CV accessions from PSI-MS ontology for standardized field extraction.

        The function handles both offset-based isolation windows (MS:1000828, MS:1000829)
        and absolute limit windows (MS:1000794, MS:1000795), computing bounds from
        offsets when absolute limits are not provided.

        See revised_mzml_schema.xsd for CV accession documentation.

        Parameters
        ----------
        mzml_path : str
            Path to the mzML file to parse.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - id: Spectrum identifier string
            - ms_level: MS level (1 for MS1, 2 for MS2, etc.)
            - scan_time: Retention time in minutes
            - polarity: 'positive' or 'negative'
            - mz: List of m/z values
            - intensity: List of intensity values
            - precursor_mz: Isolation window target m/z (MS2+ only)
            - selected_ion_mz: Selected ion m/z from selectedIonList
            - charge_state: Precursor charge state
            - isolation_window_lower_offset: Lower offset from target m/z
            - isolation_window_upper_offset: Upper offset from target m/z
            - isolation_window_lower_bound: Computed/absolute lower bound
            - isolation_window_upper_bound: Computed/absolute upper bound
            - collision_energy: Fragmentation energy
            - collision_energy_unit: Unit of collision energy
            - thermo_monoisotopic_mz: Thermo-specific monoisotopic m/z
            - user_params: List of vendor-specific parameters

        Raises
        ------
        FileNotFoundError
            If the mzML file does not exist.
        """
        if not os.path.exists(mzml_path):
            raise FileNotFoundError(f"mzML file not found: {mzml_path}")

        print(f"Reading {mzml_path} with lxml...")

        # CV Param Accessions from PSI-MS ontology
        # See revised_mzml_schema.xsd for full documentation
        CV_ACCESSIONS = {
            # Spectrum level
            "ms_level": "MS:1000511",
            "positive_scan": "MS:1000130",
            "negative_scan": "MS:1000129",
            # Scan level
            "scan_start_time": "MS:1000016",
            # Binary arrays
            "mz_array": "MS:1000514",
            "intensity_array": "MS:1000515",
            "float64": "MS:1000523",
            "float32": "MS:1000521",
            "zlib_compression": "MS:1000574",
            # Isolation window
            "isolation_target_mz": "MS:1000827",
            "isolation_lower_offset": "MS:1000828",
            "isolation_upper_offset": "MS:1000829",
            "isolation_lower_limit": "MS:1000794",
            "isolation_upper_limit": "MS:1000795",
            # Selected ion
            "selected_ion_mz": "MS:1000744",
            "charge_state": "MS:1000041",
            # Activation
            "collision_energy": "MS:1000045",
        }

        data = []

        # Use iterparse for memory-efficient parsing of large files
        context = etree.iterparse(
            mzml_path,
            events=("end",),
            tag="{http://psi.hupo.org/ms/mzml}spectrum"
        )

        for _event, spectrum in context:
            spec_id = spectrum.get("id")

            def get_cv_value(element, accession: str, dtype=str):
                """Extract value from cvParam by accession number."""
                # Why: Use local-name() to handle namespace variations across mzML files
                matches = element.xpath(
                    f".//*[local-name()='cvParam' and @accession='{accession}']"
                )
                if matches:
                    val = matches[0].get("value")
                    if val is None:
                        return None
                    try:
                        return dtype(val)
                    except (ValueError, TypeError):
                        return None
                return None

            def has_cv_param(element, accession: str) -> bool:
                """Check if a cvParam with the given accession exists."""
                matches = element.xpath(
                    f".//*[local-name()='cvParam' and @accession='{accession}']"
                )
                return len(matches) > 0

            def get_user_param_value(element, name: str, dtype=str):
                """Extract value from userParam by name."""
                matches = element.xpath(
                    f".//*[local-name()='userParam' and @name='{name}']"
                )
                if matches:
                    val = matches[0].get("value")
                    if val is None:
                        return None
                    try:
                        return dtype(val)
                    except (ValueError, TypeError):
                        return None
                return None

            # Extract spectrum-level metadata
            ms_level = get_cv_value(spectrum, CV_ACCESSIONS["ms_level"], int)
            scan_time = get_cv_value(spectrum, CV_ACCESSIONS["scan_start_time"], float)

            # Determine polarity
            polarity = None
            if has_cv_param(spectrum, CV_ACCESSIONS["positive_scan"]):
                polarity = "positive"
            elif has_cv_param(spectrum, CV_ACCESSIONS["negative_scan"]):
                polarity = "negative"

            # Thermo-specific monoisotopic m/z from userParam
            thermo_monoisotopic_mz = get_user_param_value(
                spectrum,
                "[Thermo Trailer Extra]Monoisotopic M/Z:",
                float
            )

            # Collect all userParams for extensibility
            user_params = []
            for up in spectrum.xpath(".//*[local-name()='userParam']"):
                user_params.append({
                    "name": up.get("name"),
                    "value": up.get("value"),
                    "type": up.get("type")
                })

            # Initialize precursor fields (only populated for MS2+)
            precursor_mz = None
            selected_ion_mz = None
            charge_state = None
            iso_lower_offset = None
            iso_upper_offset = None
            iso_lower_bound = None
            iso_upper_bound = None
            collision_energy = None
            collision_energy_unit = None

            # Extract precursor information for MS2+ spectra
            if ms_level is not None and ms_level > 1:
                precursors = spectrum.xpath(".//*[local-name()='precursor']")
                if precursors:
                    precursor = precursors[0]

                    # Isolation window information
                    # Why: Search within isolationWindow element specifically for accurate extraction
                    isolation_windows = precursor.xpath(
                        "./*[local-name()='isolationWindow']"
                    )
                    if isolation_windows:
                        iso_window = isolation_windows[0]
                        precursor_mz = get_cv_value(
                            iso_window,
                            CV_ACCESSIONS["isolation_target_mz"],
                            float
                        )
                        iso_lower_offset = get_cv_value(
                            iso_window,
                            CV_ACCESSIONS["isolation_lower_offset"],
                            float
                        )
                        iso_upper_offset = get_cv_value(
                            iso_window,
                            CV_ACCESSIONS["isolation_upper_offset"],
                            float
                        )
                        iso_lower_bound = get_cv_value(
                            iso_window,
                            CV_ACCESSIONS["isolation_lower_limit"],
                            float
                        )
                        iso_upper_bound = get_cv_value(
                            iso_window,
                            CV_ACCESSIONS["isolation_upper_limit"],
                            float
                        )

                    # Selected ion information
                    selected_ions = precursor.xpath(
                        ".//*[local-name()='selectedIon']"
                    )
                    if selected_ions:
                        selected_ion = selected_ions[0]
                        selected_ion_mz = get_cv_value(
                            selected_ion,
                            CV_ACCESSIONS["selected_ion_mz"],
                            float
                        )
                        charge_state = get_cv_value(
                            selected_ion,
                            CV_ACCESSIONS["charge_state"],
                            int
                        )

                    # Activation/fragmentation information
                    activations = precursor.xpath(
                        "./*[local-name()='activation']"
                    )
                    if activations:
                        activation = activations[0]
                        ce_matches = activation.xpath(
                            f".//*[local-name()='cvParam' and @accession='{CV_ACCESSIONS['collision_energy']}']"
                        )
                        if ce_matches:
                            try:
                                collision_energy = float(ce_matches[0].get("value"))
                                collision_energy_unit = ce_matches[0].get("unitName")
                            except (ValueError, TypeError):
                                pass

            # Extract binary data arrays (m/z and intensity)
            mz_array = []
            intensity_array = []

            binary_arrays = spectrum.xpath(
                ".//*[local-name()='binaryDataArrayList']/*[local-name()='binaryDataArray']"
            )

            for binary_array in binary_arrays:
                # Determine array type and encoding from cvParams
                cv_accessions = [
                    cv.get("accession")
                    for cv in binary_array.xpath("./*[local-name()='cvParam']")
                ]

                is_mz = CV_ACCESSIONS["mz_array"] in cv_accessions
                is_int = CV_ACCESSIONS["intensity_array"] in cv_accessions
                is_float64 = CV_ACCESSIONS["float64"] in cv_accessions
                is_zlib = CV_ACCESSIONS["zlib_compression"] in cv_accessions

                if is_mz or is_int:
                    binary_content = binary_array.xpath("./*[local-name()='binary']")
                    if binary_content and binary_content[0].text:
                        try:
                            decoded = base64.b64decode(binary_content[0].text)

                            if is_zlib:
                                decoded = zlib.decompress(decoded)

                            # Why: Use little-endian format as per mzML specification
                            fmt = "d" if is_float64 else "f"
                            item_size = 8 if is_float64 else 4
                            count = len(decoded) // item_size

                            array_data = struct.unpack(f"<{count}{fmt}", decoded)
                            if is_mz:
                                mz_array = list(array_data)
                            elif is_int:
                                intensity_array = list(array_data)
                        except (struct.error, zlib.error, ValueError) as e:
                            print(f"Error decoding binary data for spectrum {spec_id}: {e}")

            spec_data = {
                "id": spec_id,
                "ms_level": ms_level,
                "scan_time": scan_time,
                "polarity": polarity,
                "mz": mz_array,
                "intensity": intensity_array,
                "precursor_mz": precursor_mz,
                "selected_ion_mz": selected_ion_mz,
                "charge_state": charge_state,
                "isolation_window_lower_offset": iso_lower_offset,
                "isolation_window_upper_offset": iso_upper_offset,
                "isolation_window_lower_bound": iso_lower_bound,
                "isolation_window_upper_bound": iso_upper_bound,
                "collision_energy": collision_energy,
                "collision_energy_unit": collision_energy_unit,
                "thermo_monoisotopic_mz": thermo_monoisotopic_mz,
                "user_params": user_params,
            }

            data.append(spec_data)

            # Why: Clear processed elements to prevent memory buildup with large files
            spectrum.clear()
            while spectrum.getprevious() is not None:
                del spectrum.getparent()[0]

        schema = {
            "id": pl.String,
            "ms_level": pl.Int64,
            "scan_time": pl.Float64,
            "polarity": pl.String,
            "mz": pl.List(pl.Float64),
            "intensity": pl.List(pl.Float64),
            "precursor_mz": pl.Float64,
            "selected_ion_mz": pl.Float64,
            "charge_state": pl.Int64,
            "isolation_window_lower_offset": pl.Float64,
            "isolation_window_upper_offset": pl.Float64,
            "isolation_window_lower_bound": pl.Float64,
            "isolation_window_upper_bound": pl.Float64,
            "collision_energy": pl.Float64,
            "collision_energy_unit": pl.String,
            "thermo_monoisotopic_mz": pl.Float64,
            "user_params": pl.List(
                pl.Struct({"name": pl.String, "value": pl.String, "type": pl.String})
            ),
        }

        df = pl.DataFrame(data, schema=schema, orient="row")

        # Compute isolation window bounds from offsets when absolute limits not provided
        # Why: Some instruments provide offsets (MS:1000828/MS:1000829) instead of
        # absolute limits (MS:1000794/MS:1000795). This computes bounds for consistency.
        df = df.with_columns(
            pl.when(
                pl.col("isolation_window_lower_bound").is_null()
                & pl.col("isolation_window_lower_offset").is_not_null()
                & pl.col("precursor_mz").is_not_null()
            )
            .then(pl.col("precursor_mz") - pl.col("isolation_window_lower_offset"))
            .otherwise(pl.col("isolation_window_lower_bound"))
            .alias("isolation_window_lower_bound"),
            pl.when(
                pl.col("isolation_window_upper_bound").is_null()
                & pl.col("isolation_window_upper_offset").is_not_null()
                & pl.col("precursor_mz").is_not_null()
            )
            .then(pl.col("precursor_mz") + pl.col("isolation_window_upper_offset"))
            .otherwise(pl.col("isolation_window_upper_bound"))
            .alias("isolation_window_upper_bound"),
        )

        return df
    return (mzml_to_polars_lxml,)


@app.cell
def _(mzml_to_polars_lxml, pl, time):
    # file_path = "/home/analytit_admin/Data/raw_data/Actinomycetes/P610004-11A-001.mzML"
    # file_path = "/home/analytit_admin/Data/raw_data/iibr_data/250515_018.mzML"
    file_path = "/home/analytit_admin/Data/raw_data/ms2deepscore/Nist_LPOS_ToF18_DDA_09.mzML"
    # print("--- pymzml ---")
    # start_time = time.time() # Start timing
    # df = mzml_to_polars(file_path)
    # end_time = time.time() # End timing

    # print("Successfully converted mzML to Polars DataFrame:")
    # print(df)
    # print("\nSchema:")
    # print(df.schema)
    # print(f"\nShape: {df.shape}")
    # print(f"\nTime taken: {end_time - start_time:.2f} seconds") # Print elapsed time

    print("\n--- lxml ---")
    start_time = time.time()

    df_lxml = mzml_to_polars_lxml(file_path)
    end_time = time.time()
    print("Successfully converted mzML to Polars DataFrame (lxml):")
    # print(df_lxml.filter(pl.col("ms_level").eq(2)))
    print("\nSchema:")
    print(df_lxml.schema)
    print(df_lxml.null_count().to_init_repr())
    print(f"\nTime taken: {end_time - start_time:.2f} seconds")
    df_lxml.filter(pl.col("ms_level").eq(2))
    return


if __name__ == "__main__":
    app.run()
