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
    def mzml_to_polars_lxml(mzml_path):
        """
        Parses an mzML file using lxml and returns a Polars DataFrame.
        Extracts: scan id, ms level, mz/intensity arrays, and precursor info.
        Also extracts userParams, specifically '[Thermo Trailer Extra]Monoisotopic M/Z:'.
        """
        if not os.path.exists(mzml_path):
            raise FileNotFoundError(f"File not found: {mzml_path}")

        print(f"Reading {mzml_path} with lxml...")

        # Namespaces
        ns = "{http://psi.hupo.org/ms/mzml}"

        # CV Param Accessions
        ACC_MS_LEVEL = "MS:1000511"
        ACC_MZ_ARRAY = "MS:1000514"
        ACC_INT_ARRAY = "MS:1000515"
        ACC_FLOAT64 = "MS:1000523"
        ACC_FLOAT32 = "MS:1000521"
        ACC_ZLIB = "MS:1000574"
        ACC_NO_COMPRESSION = "MS:1000576"
        ACC_ISO_TARGET = "MS:1000827"
        ACC_ISO_LOWER_OFFSET = "MS:1000828"
        ACC_ISO_UPPER_OFFSET = "MS:1000829"
        ACC_ISO_LOWER_LIMIT = "MS:1000794"
        ACC_ISO_UPPER_LIMIT = "MS:1000795"
        ACC_SCAN_START_TIME = "MS:1000016"
        ACC_COLLISION_ENERGY = "MS:1000045"

        data = []

        context = etree.iterparse(mzml_path, events=("end",), tag=f"{ns}spectrum")

        for event, spectrum in context:
            spec_id = spectrum.get("id")

            # Get MS Level
            ms_level = None
            scan_time = None
            thermo_monoisotopic_mz = None
            user_params = []

            for cv in spectrum.findall(f"{ns}cvParam"):
                acc = cv.get("accession")
                if acc == ACC_MS_LEVEL:
                    ms_level = int(cv.get("value"))

            # Get Scan Time and User Params (usually in scanList/scan)
            scan_list = spectrum.find(f"{ns}scanList")
            if scan_list is not None:
                scan = scan_list.find(f"{ns}scan")
                if scan is not None:
                    for cv in scan.findall(f"{ns}cvParam"):
                        if cv.get("accession") == ACC_SCAN_START_TIME:
                            scan_time = float(cv.get("value"))

                    for up in scan.findall(f"{ns}userParam"):
                        name = up.get("name")
                        value = up.get("value")
                        type_ = up.get("type")
                        user_params.append({"name": name, "value": value, "type": type_})

                        if name == "[Thermo Trailer Extra]Monoisotopic M/Z:":
                            try:
                                thermo_monoisotopic_mz = float(value)
                            except (ValueError, TypeError):
                                pass

            # Precursor Info
            precursor_mz = None
            iso_lower_offset = None
            iso_upper_offset = None
            iso_lower_bound = None
            iso_upper_bound = None
            collision_energy = None
            collision_energy_unit = None

            if ms_level and ms_level > 1:
                precursor_list = spectrum.find(f"{ns}precursorList")
                if precursor_list is not None:
                    precursor = precursor_list.find(
                        f"{ns}precursor"
                    )  # Taking the first one for simplicity
                    if precursor is not None:
                        # Isolation Window
                        iso_window = precursor.find(f"{ns}isolationWindow")
                        if iso_window is not None:
                            for cv in iso_window.findall(f"{ns}cvParam"):
                                acc = cv.get("accession")
                                if acc == ACC_ISO_TARGET:
                                    precursor_mz = float(cv.get("value"))
                                elif acc == ACC_ISO_LOWER_OFFSET:
                                    iso_lower_offset = float(cv.get("value"))
                                elif acc == ACC_ISO_UPPER_OFFSET:
                                    iso_upper_offset = float(cv.get("value"))
                                elif acc == ACC_ISO_LOWER_LIMIT:
                                    iso_lower_bound = float(cv.get("value"))
                                elif acc == ACC_ISO_UPPER_LIMIT:
                                    iso_upper_bound = float(cv.get("value"))

                        # Activation
                        activation = precursor.find(f"{ns}activation")
                        if activation is not None:
                            for cv in activation.findall(f"{ns}cvParam"):
                                if cv.get("accession") == ACC_COLLISION_ENERGY:
                                    collision_energy = float(cv.get("value"))
                                    collision_energy_unit = cv.get("unitName")

            # Binary Data
            mz_array = []
            intensity_array = []

            binary_list = spectrum.find(f"{ns}binaryDataArrayList")
            if binary_list is not None:
                for binary_array in binary_list.findall(f"{ns}binaryDataArray"):
                    # Check type
                    is_mz = False
                    is_int = False
                    is_float64 = False
                    is_zlib = False

                    for cv in binary_array.findall(f"{ns}cvParam"):
                        acc = cv.get("accession")
                        if acc == ACC_MZ_ARRAY:
                            is_mz = True
                        elif acc == ACC_INT_ARRAY:
                            is_int = True
                        elif acc == ACC_FLOAT64:
                            is_float64 = True
                        elif acc == ACC_ZLIB:
                            is_zlib = True

                    if is_mz or is_int:
                        binary_content = binary_array.find(f"{ns}binary")
                        if binary_content is not None and binary_content.text:
                            decoded = base64.b64decode(binary_content.text)

                            if is_zlib:
                                try:
                                    decoded = zlib.decompress(decoded)
                                except zlib.error:
                                    pass  # Handle or log error

                            fmt = "d" if is_float64 else "f"
                            item_size = 8 if is_float64 else 4
                            count = len(decoded) // item_size

                            try:
                                array_data = struct.unpack(f"<{count}{fmt}", decoded)
                                if is_mz:
                                    mz_array = list(array_data)
                                elif is_int:
                                    intensity_array = list(array_data)
                            except struct.error:
                                print(f"Struct unpack error for spectrum {spec_id}")

            spec_data = {
                "id": spec_id,
                "ms_level": ms_level,
                "scan_time": scan_time,
                "mz": mz_array,
                "intensity": intensity_array,
                "precursor_mz": precursor_mz,
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

            # Memory management for lxml
            spectrum.clear()
            while spectrum.getprevious() is not None:
                del spectrum.getparent()[0]

        schema = {
            "id": pl.String,
            "ms_level": pl.Int64,
            "scan_time": pl.Float64,
            "mz": pl.List(pl.Float64),
            "intensity": pl.List(pl.Float64),
            "precursor_mz": pl.Float64,
            "isolation_window_lower_offset": pl.Float64,
            "isolation_window_upper_offset": pl.Float64,
            "isolation_window_lower_bound": pl.Float64,
            "isolation_window_upper_bound": pl.Float64,
            "collision_energy": pl.Float64,
            "collision_energy_unit": pl.String,
            "thermo_monoisotopic_mz": pl.Float64,
            "user_params": pl.List(pl.Struct({"name": pl.String, "value": pl.String, "type": pl.String})),
        }

        df = pl.DataFrame(data, schema=schema, orient="row")

        # Compute bounds from offsets if bounds are missing
        df = df.with_columns(
            [
                pl.when(
                    pl.col("isolation_window_lower_bound").is_null()
                    & pl.col("isolation_window_lower_offset").is_not_null()
                )
                .then(pl.col("precursor_mz") - pl.col("isolation_window_lower_offset"))
                .otherwise(pl.col("isolation_window_lower_bound"))
                .alias("isolation_window_lower_bound"),
                pl.when(
                    pl.col("isolation_window_upper_bound").is_null()
                    & pl.col("isolation_window_upper_offset").is_not_null()
                )
                .then(pl.col("precursor_mz") + pl.col("isolation_window_upper_offset"))
                .otherwise(pl.col("isolation_window_upper_bound"))
                .alias("isolation_window_upper_bound"),
            ]
        )

        return df
    return (mzml_to_polars_lxml,)


@app.cell
def _(mzml_to_polars_lxml, time):
    file_path = "/home/analytit_admin/Data/raw_data/Actinomycetes/P610004-11A-001.mzML"

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
    df_lxml
    return


if __name__ == "__main__":
    app.run()
