"""
Polars interface for parallel mass and spectrum decomposition functions.
This module provides functions that can be used with `pl.map_batches` to decompose masses
and spectra into molecular formulas using the C++ OpenMP implementations.
Each function processes a Polars DataFrame batch and returns a new DataFrame with
a `formulas` column containing the decomposition results.
"""
import polars as pl  # type: ignore
from typing import Dict, Tuple, List, Any

# Import parallel decomposition functions (with fallback if C++ not available)
from mass_decomposition_impl.mass_decomposer_cpp import (
    # Uniform-parallel functions (one bounds dict for all inputs)
    decompose_mass_parallel,
    decompose_spectra_parallel,
    decompose_spectra_known_precursor_parallel,
    # Single-call functions for per-item bounds
    decompose_mass,
    decompose_spectrum,
    decompose_spectrum_known_precursor,
)
from utils import element_data  # mapping element symbols order
num_elements = len(element_data)
element_keys = list(element_data.keys())


def decompose_mass_series(
    masses: pl.Series,
    element_bounds: Dict[str, Tuple[int, int]],
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 1000.0,
) -> pl.Series:
    """
    Decompose target masses into candidate formulas for each mass series using uniform bounds.

    This function takes a Series of precursor masses and a single bounds dict for all masses.

    Args:
        masses (pl.Series): Input series of precursor masses
        element_bounds (Dict[str, Tuple[int, int]]): mapping element symbols to (min_count, max_count)
        strategy (str): decomposition strategy ('recursive' or 'money_changing')
        tolerance_ppm (float): mass tolerance in parts per million
        min_dbe (float): minimum double bond equivalents
        max_dbe (float): maximum double bond equivalents
        max_hetero_ratio (float): maximum heteroatom-to-carbon ratio

    Returns:
        pl.Series (dtype=pl.Object): Each Series entry is a Python list of candidate formulas for the corresponding mass. 
            Each formula is represented as a dict mapping element symbols to non-zero integer counts, 
            e.g., {'C': 17, 'H': 19, 'N': 3, 'O': 1}. 
            Only elements with counts >0 appear in the dict. The list order matches the order returned by the C++ decomposer.

    Example:
        >>> import polars as pl
        >>> from polars_interface import decompose_mass_series
        >>> df = pl.DataFrame({'target_mass': [100.0, 200.0]})
        >>> bounds = {'C': (0, 10), 'H': (0, 20), 'O': (0, 5)}
        >>> result = df.with_columns(
        ...     pl.map_batches(
        ...         lambda batch: decompose_mass_series(batch['target_mass'], bounds),
        ...         pl.col('target_mass')
        ...     ).alias('formulas')
        ... )
    """
    masses_list: List[float] = masses.to_list()
    # call C++ parallel function using default strategy
    raw_results: List[List[Dict[str, int]]] = decompose_mass_parallel(
        masses_list,
        element_bounds,
        strategy=strategy,
        tolerance_ppm=tolerance_ppm,
        min_dbe=min_dbe,
        max_dbe=max_dbe,
        max_hetero_ratio=max_hetero_ratio,
    )
    # convert each formula dict into length-15 array in element_data key order
    element_keys = list(element_data.keys())
    num_elems = len(element_keys)
    arrays_list: List[List[List[int]]] = []
    for formulas in raw_results:
        formula_arrays = []
        for fdict in formulas:
            arr = [fdict.get(sym, 0) for sym in element_keys]
            formula_arrays.append(arr)
        arrays_list.append(formula_arrays)
    # return a Series of List[Array[Int64, num_elements]]
    return pl.Series(values=arrays_list,dtype=pl.List(pl.Array(pl.Int64, num_elems)))


def decompose_spectra_series(
    masses: pl.Series,
    fragments: pl.Series,
    element_bounds: Dict[str, Tuple[int, int]],
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 1000.0,
) -> pl.Series:
    """
    Decompose fragment spectra into candidate formulas for each spectrum in a Polars batch.

    This function is intended for use with `pl.map_batches`.

    Args:
        masses (pl.Series): Series of precursor masses
        fragments (pl.Series): Series of fragment mass lists
        element_bounds (Dict[str, Tuple[int, int]]): mapping element symbols to (min_count, max_count)
        strategy (str): decomposition strategy ('recursive' or 'money_changing')
        tolerance_ppm (float): mass tolerance in parts per million
        min_dbe (float): minimum double bond equivalents
        max_dbe (float): maximum double bond equivalents
        max_hetero_ratio (float): maximum heteroatom-to-carbon ratio

    Returns:
        pl.Series (dtype=pl.Object): Each Series entry is a Python list of spectrum decomposition results for the corresponding row. 
            Each result is a dict with keys:
            - 'precursor': dict of element counts for the precursor formula (non-zero counts only)
            - 'fragments': a list of lists of dicts, where each inner list corresponds to one fragment mass and contains candidate formulas (dicts)
            - 'precursor_mass': float, the precursor mass value
            - 'precursor_error_ppm': float, mass error of the precursor in ppm
            - 'fragment_masses': list of lists of floats, the actual fragment masses processed
            - 'fragment_errors_ppm': list of lists of floats, the error for each fragment mass in ppm

    Example:
        >>> df = pl.DataFrame({
        ...     'target_mass': [300.0],
        ...     'fragment_masses': [[100.0, 150.0, 200.0]]
        ... })
        >>> bounds = {'C': (0, 15), 'H': (0, 30), 'O': (0, 10)}
        >>> result = df.with_columns(
        ...     pl.map_batches(
        ...         lambda batch: decompose_spectra_series(batch['target_mass'], batch['fragment_masses'], bounds),
        ...         pl.col('target_mass'),
        ...         pl.col('fragment_masses')
        ...     ).alias('formulas')
        ... )
    """
    masses_list: List[float] = masses.to_list()
    fragments_list: List[List[float]] = fragments.to_list()
    # build spectra_data as list of (target_mass, fragment_masses)
    spectra_data = list(zip(masses_list, fragments_list))
    # call C++ parallel function using default strategy
    raw_results: List[List[Dict[str, Any]]] = decompose_spectra_parallel(
        spectra_data,
        element_bounds,
        strategy=strategy,
        tolerance_ppm=tolerance_ppm,
        min_dbe=min_dbe,
        max_dbe=max_dbe,
        max_hetero_ratio=max_hetero_ratio,
    )
    # convert fragment formula dicts to arrays
    spectra_array_list: List[List[List[List[int]]]] = []
    for spec in raw_results:
        decomp_arrays: List[List[List[int]]] = []
        for decomp in spec:
            frag_lists = decomp.get('fragments', [])
            frag_arrays: List[List[int]] = []
            for frag in frag_lists:
                # each frag is list of dicts
                arrs = [[fd.get(sym, 0) for sym in element_keys] for fd in frag]
                frag_arrays.extend(arrs)
            decomp_arrays.append(frag_arrays)
        spectra_array_list.append(decomp_arrays)
    # return list of list of arrays per spectrum
    return pl.Series(
        values=spectra_array_list,
        dtype=pl.List(pl.List(pl.Array(pl.Int64, num_elements)))
    )


def decompose_spectra_known_formula_series(
    formulas: pl.Series,
    fragments: pl.Series,
    element_bounds: Dict[str, Tuple[int, int]],
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 1000.0,
) -> pl.Series:
    """
    Decompose fragment spectra given known precursor formula arrays for each spectrum in a Polars batch.

    This function is intended for use with `pl.map_batches`.

    Args:
        formulas (pl.Series): Series of precursor formula arrays
        fragments (pl.Series): Series of fragment mass lists
        element_bounds (Dict[str, Tuple[int, int]]): mapping element symbols to (min_count, max_count)
        strategy (str): decomposition strategy ('recursive' or 'money_changing')
        tolerance_ppm (float): mass tolerance in parts per million
        min_dbe (float): minimum double bond equivalents
        max_dbe (float): maximum double bond equivalents
        max_hetero_ratio (float): maximum heteroatom-to-carbon ratio

    Returns:
        pl.Series (dtype=pl.Object): Each Series entry is a Python list of fragment decomposition results for the known precursor formula. 
            The outer list has one element per fragment mass in the input. 
            Each inner list contains dicts mapping element symbols to counts (candidate formulas) for that fragment. 
            Example: for fragments [263.142247, 220.125200], an entry might be [
                [{'C': 15, 'H': 18, 'N': 3}],  # formulas for fragment mass 263.142247
                [{'C': 14, 'H': 16, 'N': 2}]   # formulas for fragment mass 220.125200
            ]

    Example:
        >>> df = pl.DataFrame({
        ...     'precursor_formula': [[6,12,1,3,0,0,0,0,0,0,0,0,0,0,0]],  # C6H12N3...
        ...     'fragment_masses': [[100.0, 150.0]]
        ... })
        >>> bounds = {'C': (0, 15), 'H': (0, 30), 'N': (0, 5)}
        >>> result = df.with_columns(
        ...     pl.map_batches(
        ...         lambda batch: decompose_spectra_known_formula_series(batch['precursor_formula'], batch['fragment_masses'], bounds),
        ...         pl.col('precursor_formula'),
        ...         pl.col('fragment_masses')
        ...     ).alias('formulas')
        ... )
    """
    formulas_list: List[List[int]] = formulas.to_list()
    fragments_list: List[List[float]] = fragments.to_list()
    # convert precursor formula arrays to dicts
    formulas_dicts = [
        {symbol: int(count) for symbol, count in zip(element_data.keys(), arr)}
        for arr in formulas_list
    ]
    # build spectra_data as list of (precursor_formula_dict, fragment_masses)
    spectra_data = list(zip(formulas_dicts, fragments_list))
    # call C++ parallel function using default strategy
    raw_results: List[List[List[Dict[str, int]]]] = decompose_spectra_known_precursor_parallel(
        spectra_data,
        element_bounds,
        strategy=strategy,
        tolerance_ppm=tolerance_ppm,
        min_dbe=min_dbe,
        max_dbe=max_dbe,
        max_hetero_ratio=max_hetero_ratio,
    )
    # convert fragment formula dicts to arrays
    known_array_list: List[List[List[int]]] = []
    for spec in raw_results:
        frag_arrays: List[List[int]] = []
        for frag in spec:
            # frag is list of dicts
            arrs = [[fd.get(sym, 0) for sym in element_keys] for fd in frag]
            frag_arrays.extend(arrs)
        known_array_list.append(frag_arrays)
    return pl.Series(
        name='formulas',
        values=known_array_list,
        dtype=pl.List(pl.Array(pl.Int64, num_elements))
    )

# -- Per-mass bounds functions --
def decompose_mass_with_bounds_series(
    masses: pl.Series,
    bounds_series: pl.Series,
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 1000.0,
) -> pl.Series:
    """
    Decompose masses with individual bounds per mass.
    Args:
        masses: Series of precursor masses
        bounds_series: Series of dicts mapping element symbols to (min, max)
    Returns:
        Series of formula lists per mass
    """
    masses_list = masses.to_list()
    bounds_list = bounds_series.to_list()
    # per-item bounds, default strategy
    results = [
        decompose_mass(
            m, b, tolerance_ppm=tolerance_ppm,
            min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=max_hetero_ratio
        )
        for m, b in zip(masses_list, bounds_list)
    ]
    return pl.Series(name='formulas', values=results)

def decompose_spectra_with_bounds_series(
    masses: pl.Series,
    fragments: pl.Series,
    bounds_series: pl.Series,
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 1000.0,
) -> pl.Series:
    """
    Decompose fragment spectra with individual bounds per spectrum.
    Args:
        masses: Series of precursor masses
        fragments: Series of fragment mass lists
        bounds_series: Series of dicts mapping element symbols to (min, max)
    Returns:
        Series of lists of formula lists per spectrum
    """
    masses_list = masses.to_list()
    fragments_list = fragments.to_list()
    bounds_list = bounds_series.to_list()
    # per-item bounds, default strategy
    results = [
        decompose_spectrum(
            m, frag, b, tolerance_ppm=tolerance_ppm,
            min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=max_hetero_ratio
        )
        for m, frag, b in zip(masses_list, fragments_list, bounds_list)
    ]
    return pl.Series(name='formulas', values=results)

def decompose_spectra_known_formula_with_bounds_series(
    formulas: pl.Series,
    fragments: pl.Series,
    bounds_series: pl.Series,
    strategy: str = "money_changing",
    tolerance_ppm: float = 5.0,
    min_dbe: float = 0.0,
    max_dbe: float = 40.0,
    max_hetero_ratio: float = 1000.0,
) -> pl.Series:
    """
    Decompose fragment spectra with known precursor formulas and individual bounds per spectrum.
    Args:
        formulas: Series of precursor formula arrays
        fragments: Series of fragment mass lists
        bounds_series: Series of dicts mapping element symbols to (min, max)
    Returns:
        Series of lists of formula lists per spectrum
    """
    formulas_list = formulas.to_list()
    fragments_list = fragments.to_list()
    bounds_list = bounds_series.to_list()
    # convert precursor formula arrays to dicts for each spectrum
    formulas_dicts = [
        {symbol: int(count) for symbol, count in zip(element_data.keys(), arr)}
        for arr in formulas_list
    ]
    # per-item bounds, default strategy
    results = [
        decompose_spectrum_known_precursor(
            f_dict, frag, b, tolerance_ppm=tolerance_ppm,
            min_dbe=min_dbe, max_dbe=max_dbe, max_hetero_ratio=max_hetero_ratio
        )
        for f_dict, frag, b in zip(formulas_dicts, fragments_list, bounds_list)
    ]
    return pl.Series(name='formulas', values=results)

if __name__ == "__main__":
    import polars as pl  # type: ignore
    from utils import format_formula_string_to_array
    import json

    # Define common element bounds
    bounds =  {
        'C': (0, 50), 'H': (0, 100), 'N': (0, 20), 'O': (0, 40),
        'S': (0, 6), 'P': (0, 5), 'Na': (0, 0), 'F': (0, 40),
        'Cl': (0, 0), 'Br': (0, 0), 'I': (0, 4),
    }
    # 1. Mass decomposition example preserving input column
    df_mass = pl.DataFrame({'target_mass': [281.152812, 100.0]})
    out_mass = df_mass.with_columns(
        pl.struct(['target_mass'])
          .map_batches(lambda s: decompose_mass_series(
              s.struct.field('target_mass'), bounds
          ))
          .alias('precursor_formulas')
    )
    print('Mass decomposition example:')
    print(out_mass)
    # write full mass decomposition to JSON file
    with open('mass_decomposition_example.json', 'w') as f:
        json.dump(out_mass.to_dicts(), f, indent=2)

    # 2. Spectra decomposition example preserving input columns
    df_spec = pl.DataFrame({
        'target_mass': [281.152812],
        'fragment_masses': [[263.142247, 220.125200, 78.046950,321.0,79.00]]
    })
    out_spec = df_spec.with_columns(
        # precursor formulas for each target_mass
        pl.struct(['target_mass'])
          .map_batches(lambda s: decompose_mass_series(
              s.struct.field('target_mass'), bounds
          ))
          .alias('precursor_formulas'),
        # fragment formulas per spectrum
        pl.struct(['target_mass', 'fragment_masses'])
          .map_batches(lambda s: decompose_spectra_series(
              s.struct.field('target_mass'), s.struct.field('fragment_masses'), bounds
          ))
          .alias('fragment_formulas')
    )
    print('Spectra decomposition example:')
    print(out_spec)
    # write full spectra decomposition to JSON file
    with open('spectra_decomposition_example.json', 'w') as f:
        json.dump(out_spec.to_dicts(), f, indent=2)

    # 3. Known precursor spectra decomposition example
    df_known = pl.DataFrame({
        'precursor_formula': [[6,12,1,3,0,0,0,0,0,0,0,0,0,0,0]],  # C6H12N3...
        'fragment_masses': [[100.0, 150.0]]
    })
    out_known = df_known.with_columns(
        # fragment formulas per spectrum using known precursor formula
        pl.struct(['precursor_formula', 'fragment_masses'])
          .map_batches(lambda s: decompose_spectra_known_formula_series(
              s.struct.field('precursor_formula'), s.struct.field('fragment_masses'), bounds
          ))
          .alias('fragment_formulas')
    )
    print('Known precursor spectra decomposition example:')
    print(out_known)
    # write full known precursor spectra decomposition to JSON file
    with open('spectra_known_precursor_example.json', 'w') as f:
        json.dump(out_known.to_dicts(), f, indent=2)


