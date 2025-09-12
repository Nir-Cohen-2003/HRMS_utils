from hrms_utils.rdkit import (
    sanitize_smiles_polars,
    sanitize_smiles,
    inchi_to_smiles_polars,
    inchi_to_smiles_list,
    smiles_to_inchi_polars,
    smiles_to_inchi_list,
    smiles_to_inchikey_polars,
    smiles_to_inchikey_list,
)
import polars as pl
import time

smiles_list = [
    # very small / simple
    "C",                    # methane
    "O",                    # water / hydroxyl
    "CC",                   # ethane
    "CCO",                  # ethanol
    "CC(=O)O",              # acetic acid

    # aromatics and simple heterocycles
    "c1ccccc1",             # benzene (aromatic lowercase)
    "c1ccccc1O",            # phenol
    "c1ccccc1N",            # aniline
    "c1ccccc1C(=O)O",       # benzoic acid
    "C1=CC=CC=C1",          # benzene (explicit)
    "C1=CC=CN=C1",          # pyridine-like (heterocycle)

    # common drugs / drug-like molecules
    "CC(=O)OC1=CC=CC=C1C(=O)O",                       # aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",                  # ibuprofen (drug-like)
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                     # caffeine
    "C[C@H](N)C(=O)O",                                # L-alanine (chiral)
    "NCC(=O)O",                                       # glycine (simple amino acid)
    "CCN(CC)CC",                                      # triethylamine fragment (drug-like amine)
    "C[N+](C)(C)C",                                   # tetramethylammonium (charged)

    # larger / lipophilic / polymer-like
    "CCCCCCCCCCCCCCCC",      # long alkane chain (C16)
    "C1CCCCC1O",             # cyclohexanol
    "C1=CC2=C(C=C1)C=CC(=C2)O",  # naphthol-like polycycle

    # peptides / polar molecules
    "CC(=O)NCC(=O)O",        # dipeptide-like fragment
    "N[C@@H](CC1=CN=CN1)C(=O)O",  # amino acid with heterocycle (chiral)

    # isotopically labelled / ionic
    "[13CH4]",               # isotopic methane
    "CC(=O)[O-]",            # acetate anion
    "C[N+](C)(C)C",          # quaternary ammonium

    # substituted aromatics
    "C1=CC=CC=C1C(C)(C)C",   # tert-butylbenzene
    "C1=CC=CC=C1C(=O)OCC",   # ethyl benzoate

    # intentionally malformed / invalid examples (various failure modes)
    "clc1ccccc1",            # wrong case for Cl (should be 'Cl')
    "C1CCCCC",               # unclosed ring (missing '1' closure)
    "C(C(=O)O",              # unbalanced parentheses
    "C1=CC=CC=C1N(",         # incomplete branch
    "C1=CC=CC=C1IO",         # likely a typo (I followed by O)
    "Xx",                    # unknown element symbol
    "",                      # empty string
    " ",                     # whitespace only
    "C C O",                 # spaces in SMILES (invalid formatting)
    "C[N+]",                 # incomplete charged atom specification
    "C@@@",                  # invalid chirality token
]



def python_test(multiplier: int)-> None:
    """
    Test the pure-Python list-based functions in hrms_utils.rdkit.mol.py and
    produce simple timing/benchmark output.

    - multiplier: multiplies the smiles_list to create a larger input for benchmarking.
    """
    assert isinstance(multiplier, int) and multiplier > 0, "multiplier must be a positive integer"

    test_smiles = smiles_list * multiplier
    n = len(test_smiles)
    print(f"Running python_test with {n} SMILES (multiplier={multiplier})")

    # sanitize_smiles (list-based)
    t0 = time.perf_counter()
    sanitized = sanitize_smiles(test_smiles, batch_size=1024)
    t1 = time.perf_counter()
    assert isinstance(sanitized, list), "sanitize_smiles must return a list"
    assert len(sanitized) == n, "sanitize_smiles must preserve input length"
    # basic sanity: all outputs are strings
    assert all(isinstance(x, str) for x in sanitized), "All sanitized entries must be strings"
    print(f"sanitize_smiles: {t1-t0:.4f}s total, {(t1-t0)/n:.6f}s per item")

    # smiles_to_inchi_list
    t0 = time.perf_counter()
    inchi_list = smiles_to_inchi_list(test_smiles, batch_size=1024)
    t1 = time.perf_counter()
    assert isinstance(inchi_list, list)
    assert len(inchi_list) == n
    assert all(isinstance(x, str) for x in inchi_list)
    print(f"smiles_to_inchi_list: {t1-t0:.4f}s total, {(t1-t0)/n:.6f}s per item")

    # smiles_to_inchikey_list
    t0 = time.perf_counter()
    inchikey_list = smiles_to_inchikey_list(test_smiles, batch_size=1024)
    t1 = time.perf_counter()
    assert isinstance(inchikey_list, list)
    assert len(inchikey_list) == n
    assert all(isinstance(x, str) for x in inchikey_list)
    print(f"smiles_to_inchikey_list: {t1-t0:.4f}s total, {(t1-t0)/n:.6f}s per item")

    # spot-check: valid simple SMILES should not produce empty string after sanitization
    # Why: ensure canonicalization doesn't drop simple valid structures.
    try:
        idx_c = test_smiles.index("C")
        assert sanitized[idx_c] != "", "Valid SMILES 'C' should not sanitize to empty string"
    except ValueError:
        # if 'C' not present in multiplied list, skip the check
        pass

    print("python_test completed successfully.")


def polars_test(multiplier: int)-> None:
    """
    Test the 'polars' wrappers and polars return types. Benchmark timings.

    - multiplier: multiplies the smiles_list to create a larger input for benchmarking.
    """
    assert isinstance(multiplier, int) and multiplier > 0, "multiplier must be a positive integer"

    test_smiles = smiles_list * multiplier
    n = len(test_smiles)
    print(f"Running polars_test with {n} SMILES (multiplier={multiplier})")

    # sanitize_smiles_polars expects a pl.Series
    series_in = pl.Series("smiles", test_smiles, dtype=pl.Utf8)
    t0 = time.perf_counter()
    sanitized_series = sanitize_smiles_polars(series_in, batch_size=1024)
    t1 = time.perf_counter()
    assert isinstance(sanitized_series, pl.Series)
    assert sanitized_series.len() == n
    assert sanitized_series.dtype == pl.Utf8
    print(f"sanitize_smiles_polars: {t1-t0:.4f}s total, {(t1-t0)/n:.6f}s per item")

    # smiles_to_inchi_polars (returns a pl.Series)
    t0 = time.perf_counter()
    inchi_series = smiles_to_inchi_polars(test_smiles, batch_size=1024)
    t1 = time.perf_counter()
    assert isinstance(inchi_series, pl.Series)
    assert inchi_series.len() == n
    print(f"smiles_to_inchi_polars: {t1-t0:.4f}s total, {(t1-t0)/n:.6f}s per item")

    # smiles_to_inchikey_polars (returns a pl.Series)
    t0 = time.perf_counter()
    inchikey_series = smiles_to_inchikey_polars(test_smiles, batch_size=1024)
    t1 = time.perf_counter()
    assert isinstance(inchikey_series, pl.Series)
    assert inchikey_series.len() == n
    print(f"smiles_to_inchikey_polars: {t1-t0:.4f}s total, {(t1-t0)/n:.6f}s per item")

    # basic content checks: all entries are strings in the Series
    assert all(isinstance(x, str) for x in sanitized_series.to_list())
    assert all(isinstance(x, str) for x in inchi_series.to_list())
    assert all(isinstance(x, str) for x in inchikey_series.to_list())

    print("polars_test completed successfully.")


if __name__ == "__main__":
    # get the multiplier from command line args if provided
    import sys
    multiplier = 1000  # default
    if len(sys.argv) > 1:
        try:
            multiplier = int(sys.argv[1])
            if multiplier <= 0:
                raise ValueError
        except ValueError:
            print("Usage: python mol.py [multiplier]")
            print("multiplier must be a positive integer.")
            sys.exit(1)
    print(f"Using multiplier={multiplier}")
    python_test(multiplier)
    print()
    polars_test(multiplier)
