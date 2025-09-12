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
import timeit
from typing import Callable, Tuple, Any, List
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

# Why: provide a parallel test vector of InChI strings for conversion/round-trip tests.
# For entries that are intentionally invalid SMILES, we provide a similarly invalid InChI
# token (prefixed with "InChI=INVALID:...") so tests can assert failure modes without using
# empty strings which can be ambiguous.
inchi_list = [
    "InChI=1S/CH4/h1H4",  # methane
    "InChI=1S/H2O/h1H2",  # water
    "InChI=1S/C2H6/c1-2/h1-2H3",  # ethane
    "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # ethanol
    "InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)",  # acetic acid

    "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",  # benzene
    "InChI=1S/C6H6O/c7-6-4-2-1-3-5-6/h1-5,7H",  # phenol
    "InChI=1S/C6H7N/c1-2-4-6-5-3-1/h1-6H,7H",  # aniline
    "InChI=1S/C7H6O2/c8-7(9)6-4-2-1-3-5-6/h1-5H,(H,8,9)",  # benzoic acid
    "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",  # benzene (explicit)
    "InChI=1S/C5H5N/c1-2-4-6-5-3-1/h1-5H",  # pyridine

    "InChI=1S/C9H8O4/c1-6(10)13-7-4-2-3-5-8(7)9(11)12/h2-5H,1H3,(H,11,12)",  # aspirin
    "InChI=1S/C13H18O2/c1-9(2)10-6-7-11(8-10)12(3)13(14)15/h6-9,12H,1-3H3,(H,14,15)",  # ibuprofen
    "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(2)6/h1-3H3",  # caffeine
    "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3/t2-/m0/s1",  # L-alanine (with stereochemistry)
    "InChI=1S/C2H5NO2/c3-1-2(4)5/h1,3H2,(H,4,5)",  # glycine

    "InChI=1S/C6H15N/c1-4-7(5-2)6-3/h4-6H2,1-3H3",  # triethylamine
    "InChI=1S/C4H12N/c1-5(2,3)4/h1-4H3/q+1",  # tetramethylammonium cation

    "InChI=1S/C16H34/c1-3-5-7-9-11-13-15-16-14-12-10-8-6-4-2/h3-15H2,1-2H3",  # hexadecane (C16)
    "InChI=1S/C6H12O/c7-6-4-2-1-3-5-6/h6-7H,1-5H2",  # cyclohexanol
    "InChI=1S/C10H8O/c11-10-8-6-4-2-1-3-5-7-9-8/h1-7,11H",  # naphthol-like

    "InChI=1S/C4H7NO3/c1-4(6)5-2-3-7/h2-3H2,1H3,(H,5,7)",  # dipeptide-like fragment (approx)
    "InChI=1S/C6H8N4O2/c1-4-6(9-7-2-3-8-9)5(4)10/h1-3H2,(H,7,8)",  # heterocyclic amino acid (approx)

    "InChI=1S/[13CH4]/h1H4",  # isotopically labelled methane (13C)
    "InChI=1S/C2H3O2/c1-2(3)4/h1H3/p-1",  # acetate anion (deprotonated)
    "InChI=1S/C4H12N/c1-5(2,3)4/h1-4H3/q+1",  # quaternary ammonium (same as tetramethylammonium)

    "InChI=1S/C10H14/c1-10(2,3)9-7-5-4-6-8-9/h4-8H,1-3H3",  # tert-butylbenzene
    "InChI=1S/C9H10O2/c1-2-11-9-7-5-3-4-6-8-9/h3-8H,2H2,1H3",  # ethyl benzoate

    # Invalid / intentionally malformed inputs — provide similarly malformed InChI tokens
    "InChI=INVALID:clc1ccccc1",            # wrong case for Cl -> invalid InChI token
    "InChI=INVALID:unclosed_ring",         # unclosed ring
    "InChI=INVALID:unbalanced_parentheses",# unbalanced parentheses
    "InChI=INVALID:incomplete_branch",     # incomplete branch
    "InChI=INVALID:typo_IO",               # probable typo (I followed by O)
    "InChI=INVALID:unknown_element",       # unknown element symbol
    "InChI=INVALID:empty",                 # empty SMILES -> invalid InChI token
    "InChI=INVALID:whitespace",            # whitespace-only SMILES
    "InChI=INVALID:spaces_in_smiles",      # spaces in SMILES formatting
    "InChI=INVALID:incomplete_charge",     # incomplete charged atom specification
    "InChI=INVALID:invalid_chirality_token",# invalid chirality token
]
# ...existing code.

def _benchmark_min_time(func: Callable[..., Any], *args, runs: int = 5, **kwargs) -> Tuple[Any, float]:
    """
    Use timeit.repeat to measure execution times and return (result, min_duration_seconds).

    Why:
    - timeit.repeat is designed to measure small snippets robustly and reduces measurement noise.
    - We run the callable 'runs' times (number=1 each) and take the minimum observed duration.
    - We execute the function once more to capture and return its result (this ensures callers get
      a real function return value while the timing is derived from the repeat runs).
    Note: the returned result may not correspond to the fastest measured run (timing and captured
    result are separated to allow using Timer.repeat for robust timing).
    """
    assert isinstance(runs, int) and runs > 0, "runs must be a positive integer"
    timer = timeit.Timer(lambda: func(*args, **kwargs))
    times = timer.repeat(repeat=runs, number=1)
    min_time = float(min(times))
    # Run once to obtain the actual result; fail fast if the function errors.
    result = func(*args, **kwargs)
    return result, min_time


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
    sanitized, sanitize_time = _benchmark_min_time(sanitize_smiles, test_smiles, batch_size=BATCH_SIZE, runs=5)
    assert isinstance(sanitized, list), "sanitize_smiles must return a list"
    assert len(sanitized) == n, "sanitize_smiles must preserve input length"
    # basic sanity: all outputs are strings
    assert all(isinstance(x, str) for x in sanitized), "All sanitized entries must be strings"
    print(f"sanitize_smiles: {sanitize_time:.4f}s total, {sanitize_time/n:.6f}s per item")

    # smiles_to_inchi_list
    inchi_list, inchi_time = _benchmark_min_time(smiles_to_inchi_list, test_smiles, batch_size=BATCH_SIZE, runs=5)
    assert isinstance(inchi_list, list)
    assert len(inchi_list) == n
    assert all(isinstance(x, str) for x in inchi_list)
    print(f"smiles_to_inchi_list: {inchi_time:.4f}s total, {inchi_time/n:.6f}s per item")

    # smiles_to_inchikey_list
    inchikey_list, inchikey_time = _benchmark_min_time(smiles_to_inchikey_list, test_smiles, batch_size=BATCH_SIZE, runs=5)
    assert isinstance(inchikey_list, list)
    assert len(inchikey_list) == n
    assert all(isinstance(x, str) for x in inchikey_list)
    print(f"smiles_to_inchikey_list: {inchikey_time:.4f}s total, {inchikey_time/n:.6f}s per item")

    # inchi_to_smiles_list (test InChI -> SMILES list-based converter)
    test_inchis = inchi_list * multiplier
    n_inchis = len(test_inchis)
    inchi_to_smiles_results, inchi_to_smiles_time = _benchmark_min_time(inchi_to_smiles_list, test_inchis, batch_size=BATCH_SIZE, runs=5)
    assert isinstance(inchi_to_smiles_results, list), "inchi_to_smiles_list must return a list"
    assert len(inchi_to_smiles_results) == n_inchis, "inchi_to_smiles_list must preserve input length"
    assert all(isinstance(x, str) for x in inchi_to_smiles_results), "All inchi->smiles outputs must be strings"
    print(f"inchi_to_smiles_list: {inchi_to_smiles_time:.4f}s total, {inchi_to_smiles_time/n_inchis:.6f}s per item")

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
    sanitized_series, sanitize_series_time = _benchmark_min_time(sanitize_smiles_polars, series_in, batch_size=BATCH_SIZE, runs=5)
    assert isinstance(sanitized_series, pl.Series)
    assert sanitized_series.len() == n
    assert sanitized_series.dtype == pl.Utf8
    print(f"sanitize_smiles_polars: {sanitize_series_time:.4f}s total, {sanitize_series_time/n:.6f}s per item")

    # smiles_to_inchi_polars (returns a pl.Series)
    inchi_series, inchi_series_time = _benchmark_min_time(smiles_to_inchi_polars, test_smiles, batch_size=BATCH_SIZE, runs=5)
    assert isinstance(inchi_series, pl.Series)
    assert inchi_series.len() == n
    print(f"smiles_to_inchi_polars: {inchi_series_time:.4f}s total, {inchi_series_time/n:.6f}s per item")

    # smiles_to_inchikey_polars (returns a pl.Series)
    inchikey_series, inchikey_series_time = _benchmark_min_time(smiles_to_inchikey_polars, test_smiles, batch_size=BATCH_SIZE, runs=5)
    assert isinstance(inchikey_series, pl.Series)
    assert inchikey_series.len() == n
    print(f"smiles_to_inchikey_polars: {inchikey_series_time:.4f}s total, {inchikey_series_time/n:.6f}s per item")

    # inchi_to_smiles_polars (test InChI -> SMILES polars wrapper)
    test_inchis = inchi_list * multiplier
    series_inchi = pl.Series("inchi", test_inchis, dtype=pl.Utf8)
    inchi_to_smiles_series, inchi_to_smiles_series_time = _benchmark_min_time(inchi_to_smiles_polars, series_inchi, batch_size=BATCH_SIZE, runs=5)
    assert isinstance(inchi_to_smiles_series, pl.Series), "inchi_to_smiles_polars must return a pl.Series"
    assert inchi_to_smiles_series.len() == series_inchi.len(), "inchi_to_smiles_polars must preserve Series length"
    assert inchi_to_smiles_series.dtype == pl.Utf8
    print(f"inchi_to_smiles_polars: {inchi_to_smiles_series_time:.4f}s total, {inchi_to_smiles_series_time/series_inchi.len():.6f}s per item")

    # basic content checks: all entries are strings in the Series
    assert all(isinstance(x, str) for x in sanitized_series.to_list())
    assert all(isinstance(x, str) for x in inchi_series.to_list())
    assert all(isinstance(x, str) for x in inchikey_series.to_list())
    assert all(isinstance(x, str) for x in inchi_to_smiles_series.to_list())

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
    BATCH_SIZE = 5000  # smaller batch size for testing; adjust as needed
    print(f"Using BATCH_SIZE={BATCH_SIZE}")
    polars_test(multiplier)
    print()
    python_test(multiplier)
