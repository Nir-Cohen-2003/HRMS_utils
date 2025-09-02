"""
Single source of truth for formula element definitions used in Python code.

- Only the elements listed here are supported by formula functions.
- The order is fixed and must not be changed; it is by increasing monoisotopic mass.
- All formula arrays, element lookups, and mass calculations must use this order.
- Do not add, remove, or reorder elements without updating all dependent code.
"""

from typing import NamedTuple, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class IsotopicDistribution:
    mass_differences: Tuple[float, ...]         # Mass differences from base isotope (M+1, M+2, etc.)
    abundances: Tuple[float, ...]               # Relative abundances for each isotope (sum to ~1)
    isotope_symbols: Tuple[str, ...]            # Isotope symbols (e.g., "12C", "13C")

class ElementInfo(NamedTuple):
    symbol: str
    mass: float  # Monoisotopic mass
    regex: str   # Regex for extracting element count from formula string
    isotope: str # Most abundant isotope symbol (may be same as element)
    isotope_mass_diff: float  # Mass difference to base isotope (0 if not relevant)
    isotopic_distribution: Optional[IsotopicDistribution] # None if not relevant

# Ordered by monoisotopic mass (lowest to highest)
ELEMENTS: Tuple[ElementInfo, ...] = (
    ElementInfo('H',   1.007825,    r'H(\d+|[A-Z]|$){1}',    '1H',   0.0, None),
    ElementInfo('B',  11.009305,    r'B(\d+|[A-Z]|$){1}',   '11B',   0.0, None),
    ElementInfo('C',  12.000000,    r'C(\d+|[A-Z]|$){1}',   '12C',   0.0,
        IsotopicDistribution(
            mass_differences=(1.003355,), 
            abundances=(0.9893, 0.0107), 
            isotope_symbols=("12C", "13C")
        )
    ),
    ElementInfo('N',  14.003074,    r'N(\d+|[A-Z]|$){1}',   '14N',   0.0,
        IsotopicDistribution(
            mass_differences=(0.997035,), 
            abundances=(0.996, 0.004), 
            isotope_symbols=("14N", "15N")
        )
    ),
    ElementInfo('O',  15.994915,    r'O(\d+|[A-Z]|$){1}',   '16O',   0.0, None),
    ElementInfo('F',  18.998403,    r'F(\d+|[A-Z]|$){1}',   '19F',   0.0, None),
    ElementInfo('Na', 22.989770,    r'Na(\d+|[A-Z]|$){1}',  '23Na',  0.0, None),
    ElementInfo('Si', 27.9769265,   r'Si(\d+|[A-Z]|$){1}',  '28Si',  0.0, None),
    ElementInfo('P',  30.973762,    r'P(\d+|[A-Z]|$){1}',   '31P',   0.0, None),
    ElementInfo('S',  31.972071,    r'S(\d+|[A-Z]|$){1}',   '32S',   0.0,
        IsotopicDistribution(
            mass_differences=(1.995796,), 
            abundances=(0.9493, 0.0429), 
            isotope_symbols=("32S", "34S")
        )
    ),
    ElementInfo('Cl', 34.96885271,  r'Cl(\d+|[A-Z]|$){1}',  '35Cl',  1.99705,
        IsotopicDistribution(
            mass_differences=(1.99705,), 
            abundances=(0.7578, 0.2422), 
            isotope_symbols=("35Cl", "37Cl")
        )
    ),
    ElementInfo('K',  38.963707,    r'K(\d+|[A-Z]|$){1}',   '39K',   1.99820, None),
    ElementInfo('As', 74.921596,    r'As(\d+|[A-Z]|$){1}',  '75As',  0.0, None),
    ElementInfo('Br', 78.918338,    r'Br(\d+|[A-Z]|$){1}',  '79Br',  1.99795,
        IsotopicDistribution(
            mass_differences=(1.99795,), 
            abundances=(0.5069, 0.4931), 
            isotope_symbols=("79Br", "81Br")
        )
    ),
    ElementInfo('I', 126.904468,    r'I(\d+|[A-Z]|$){1}',   '127I',  0.0, None),
)

NUM_ELEMENTS: int = len(ELEMENTS)
ELEMENT_SYMBOLS: Tuple[str, ...] = tuple(e.symbol for e in ELEMENTS)
ELEMENT_MASSES: Tuple[float, ...] = tuple(e.mass for e in ELEMENTS)
ELEMENT_REGEXES: Tuple[str, ...] = tuple(e.regex for e in ELEMENTS)
ELEMENT_ISOTOPES: Tuple[str, ...] = tuple(e.isotope for e in ELEMENTS)
ELEMENT_ISOTOPE_MASS_DIFFS: Tuple[float, ...] = tuple(e.isotope_mass_diff for e in ELEMENTS)
ELEMENT_ISOTOPIC_DISTRIBUTIONS: Tuple[Optional[IsotopicDistribution], ...] = tuple(e.isotopic_distribution for e in ELEMENTS)
DEFAULT_MIN_BOUND = {symbol: 0 for symbol in ELEMENT_SYMBOLS}
DEFAULT_MAX_BOUND = {
    'H': 100,
    'B': 0,
    'C': 50,
    'N': 20,
    'O': 20,
    'F': 40,
    'Na': 0,
    'Si': 0,
    'P': 5,
    'S': 5,
    'Cl': 10,
    'K': 0,
    'As': 0,
    'Br': 10,
    'I': 5,
}

# For fast lookup by symbol
ELEMENT_INDEX: Dict[str, int] = {e.symbol: i for i, e in enumerate(ELEMENTS)}

def validate_formula_array(arr):
    """Check that a formula array is valid for this element table."""
    if not hasattr(arr, "__len__") or len(arr) != NUM_ELEMENTS:
        raise ValueError(f"Formula array must have length {NUM_ELEMENTS} (got {len(arr)})")
    if any(x < 0 for x in arr):
        raise ValueError("Formula array contains negative element counts")
    return True

