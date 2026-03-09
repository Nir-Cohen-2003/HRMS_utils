"""
Check the spectral library conversion pipeline using dataframely for schema validation.
"""
import argparse
from pathlib import Path
import polars as pl
import dataframely as dfm
from time import perf_counter

from hrms_utils.formats.spectral_library import (
    _parse_file,
    _process_pipeline,
    _filter_invalid_entries,
    _enrich_with_pubchem,
    _deduplicate_spectra,
    _standardize_structures
)

class RawParsedSchema(dfm.Schema):
    raw_spectrum_mz = dfm.List(dfm.Float64(), nullable=True)
    raw_spectrum_intensity = dfm.List(dfm.Float64(), nullable=True)

class ProcessedPipelineSchema(dfm.Schema):
    precursor_mz = dfm.Float64(nullable=True)
    cleaned_normalized_mz = dfm.List(dfm.Float64(), nullable=True)
    cleaned_normalized_intensity = dfm.List(dfm.Float64(), nullable=True)

class StandardizedSchema(dfm.Schema):
    base_inchikey = dfm.String(nullable=True)
    smiles = dfm.String(nullable=True)
    inchi = dfm.String(nullable=True)

def check_schemas(input_dir: Path, pubchem_path: Path):
    print(f"Checking library conversion for {input_dir}")
    files = list(input_dir.glob("*.msp")) + list(input_dir.glob("*.mspec")) + list(input_dir.glob("*.mgf"))
    if not files:
        print("No files found!")
        return
        
    f = Path("tests/data/msp_sample.msp")
    print(f"Parsing {f}...")
    
    t0 = perf_counter()
    lf = _parse_file(f, includes_MSn=False)
    df = lf.collect(engine="streaming")
    print(f"[{perf_counter()-t0:.2f}s] Checking RawParsedSchema...")
    assert RawParsedSchema.is_valid(df), "Raw parsing schema validation failed!"
    
    t1 = perf_counter()
    lf = df.lazy().with_columns(pl.lit(f.name).alias("source_file"))
    lf = _process_pipeline(lf, 10.0, 5.0, 5.0)
    df = lf.collect(engine="streaming")
    print(f"[{perf_counter()-t1:.2f}s] Checking ProcessedPipelineSchema...")
    assert ProcessedPipelineSchema.is_valid(df), "Process pipeline schema validation failed!"
    
    t2 = perf_counter()
    lf = _filter_invalid_entries(df.lazy())
    if pubchem_path and pubchem_path.exists():
        lf = _enrich_with_pubchem(lf, pubchem_path)
    df = lf.collect(engine="streaming")
    print(f"[{perf_counter()-t2:.2f}s] Collected after enrichment")

    t3 = perf_counter()
    df = _deduplicate_spectra(df.lazy(), fragment_tolerance_ppm=5.0, molecular_ion_tolerance_ppm=5.0, threshold=0.99).collect(engine="streaming")
    print(f"[{perf_counter()-t3:.2f}s] Collected after deduplication")

    t4 = perf_counter()
    df = _standardize_structures(df)
    print(f"[{perf_counter()-t4:.2f}s] Standardized structures")
    
    print("Checking StandardizedSchema...", end=" ")
    assert StandardizedSchema.is_valid(df), "Standardized schema validation failed!"
    print("OK")
    print("All schema checks passed successfully!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--pubchem", "-p", type=Path, default=None)
    args = parser.parse_args()
    
    check_schemas(args.input_path, args.pubchem)

if __name__ == "__main__":
    main()
