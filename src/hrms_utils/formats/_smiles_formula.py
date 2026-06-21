"""Fallback implementation of ``smiles_to_formula`` for the spectral library pipeline.

The ``parallel_rdkit`` package is expected to expose a C++/OpenMP-backed
``smiles_to_formula`` function. When that function is not available, this
module provides a pure-RDKit implementation that follows the same contract
documented in ``parallel_rdkit_docs.md``:

* Returns element counts in the fixed order
  ``[H, C, N, O, F, Na, P, S, Cl, K, Br, I]``.
* For a ``polars.Series`` input, returns a ``pl.Series`` of dtype
  ``Array(Int64, shape=(12,))``.
* For an iterable input, returns a 2D ``numpy.ndarray`` of shape ``(n, 12)``.
* Invalid SMILES produce an all-zero row. Atoms whose elements are not in the
  supported 12-element list are silently ignored.
"""

from __future__ import annotations

import re
from typing import Iterable, Union

import numpy as np
import polars as pl

# Keep these in sync with ``hrms_utils.formula_annotation.element_table.ELEMENTS``.
_ELEMENT_SYMBOLS: tuple[str, ...] = (
    "H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "K", "Br", "I",
)
_NUM_ELEMENTS: int = len(_ELEMENT_SYMBOLS)
_SYMBOL_TO_INDEX: dict[str, int] = {sym: i for i, sym in enumerate(_ELEMENT_SYMBOLS)}
_DTYPE: type = np.int64


def _formula_string_to_array(formula: str) -> np.ndarray:
    """Parse a molecular formula string into an element-count array."""
    arr = np.zeros(_NUM_ELEMENTS, dtype=_DTYPE)
    if not formula:
        return arr
    # rdkit returns Hill-system formulas like "C2H6O", "CH4", "C2H4O2S".
    # The regex captures the (optional) digit count following each symbol.
    import re

    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    for match in pattern.finditer(formula):
        symbol = match.group(1)
        count_str = match.group(2)
        idx = _SYMBOL_TO_INDEX.get(symbol)
        if idx is None:
            # Atom element is not in the supported 12-element table; ignore it.
            continue
        arr[idx] = int(count_str) if count_str else 1
    return arr


def _smiles_to_array(smiles: str | None) -> np.ndarray:
    """Compute the element-count array for a single SMILES string."""
    arr = np.zeros(_NUM_ELEMENTS, dtype=_DTYPE)
    if not smiles:
        return arr
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return arr
    if mol is None:
        return arr
    formula = CalcMolFormula(mol)
    return _formula_string_to_array(formula)


def smiles_to_formula(
    smiles: Union[Iterable[str], pl.Series],
) -> Union[np.ndarray, pl.Series]:
    """Compute molecular formulas for a list or Polars Series of SMILES strings.

    Parameters
    ----------
    smiles:
        An iterable of SMILES strings, or a ``polars.Series`` of SMILES.

    Returns
    -------
    ``numpy.ndarray`` of shape ``(n, 12)`` for list/iterable input, or a
    ``polars.Series`` of dtype ``Array(Int64, shape=(12,))`` for a Polars
    Series input. Element counts follow the order
    ``[H, C, N, O, F, Na, P, S, Cl, K, Br, I]``. Invalid SMILES produce a row
    of zeros.
    """
    if isinstance(smiles, pl.Series):
        smiles_list = smiles.to_list()
        result = np.zeros((len(smiles_list), _NUM_ELEMENTS), dtype=_DTYPE)
        for i, s in enumerate(smiles_list):
            result[i] = _smiles_to_array(s)
        return pl.Series(result)
    smiles_list = list(smiles)
    result = np.zeros((len(smiles_list), _NUM_ELEMENTS), dtype=_DTYPE)
    for i, s in enumerate(smiles_list):
        result[i] = _smiles_to_array(s)
    return result
