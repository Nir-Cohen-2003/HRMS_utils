# cython: language_level=3, boundscheck=False, wraparound=False
"""
High-performance Cython implementation of the SIRIUS mass decomposition algorithm with chemical constraints.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from typing import List, Dict, Tuple, Optional

# Standard atomic masses (most abundant isotopes)
ATOMIC_MASSES = {
    'C': 12.0000000,
    'H': 1.0078250,
    'N': 14.0030740,
    'O': 15.9949146,
    'P': 30.9737620,
    'S': 31.9720718,
    'F': 18.9984032,
    'Cl': 34.9688527,
    'Br': 78.9183376,
    'I': 126.9044719,
    'Si': 27.9769271,
    'Na': 22.9897693,
    'K': 38.9637069,
    'Ca': 39.9625912,
    'Mg': 23.9850423,
    'Fe': 55.9349421,
    'Zn': 63.9291466,
    'Se': 79.9165218,
    'B': 11.0093054,
    'Al': 26.9815386
}

cdef struct Element:
    char symbol[4]  # Element symbol (max 3 chars + null terminator)
    double mass     # Atomic mass
    int min_count   # Minimum count
    int max_count   # Maximum count

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CythonSiriusDecomposer:
    """
    Ultra-fast Cython implementation of SIRIUS mass decomposition algorithm with chemical constraints.
    """
    
    cdef Element* elements
    cdef int n_elements
    cdef double* min_residues
    cdef double* max_residues
    cdef double target_mass
    cdef double tolerance
    cdef int max_results
    cdef list results
    
    # Chemical constraint parameters
    cdef double min_dbe
    cdef double max_dbe
    cdef double max_hetero_ratio
    cdef bint use_dbe_constraint
    cdef bint use_hetero_constraint
    
    # Element indices for fast constraint checking
    cdef int c_idx, h_idx, n_idx, p_idx
    cdef int f_idx, cl_idx, br_idx, i_idx
    
    def __cinit__(self, element_bounds: Dict[str, Tuple[int, int]], 
                  double target_mass, double tolerance_ppm = 5.0, 
                  int max_results = 1000000,
                  double min_dbe = -1000.0, double max_dbe = 1000.0,
                  double max_hetero_ratio = 1000.0):
        """
        Initialize the mass decomposer with chemical constraints.
        
        Args:
            element_bounds: Dict mapping element symbols to (min_count, max_count)
            target_mass: Target mass to decompose
            tolerance_ppm: Mass tolerance in parts per million
            max_results: Maximum number of results to return
            min_dbe: Minimum double bond equivalents (default: no constraint)
            max_dbe: Maximum double bond equivalents (default: no constraint)
            max_hetero_ratio: Maximum ratio of heteroatoms to carbons (default: no constraint)
        """
        self.target_mass = target_mass
        self.tolerance = target_mass * tolerance_ppm / 1e6
        self.max_results = max_results
        self.results = []
        
        # Chemical constraint parameters
        self.min_dbe = min_dbe
        self.max_dbe = max_dbe
        self.max_hetero_ratio = max_hetero_ratio
        self.use_dbe_constraint = (min_dbe > -999.0 or max_dbe < 999.0)
        self.use_hetero_constraint = (max_hetero_ratio < 999.0)
        
        # Initialize element indices for constraint checking
        self.c_idx = -1
        self.h_idx = -1
        self.n_idx = -1
        self.p_idx = -1
        self.f_idx = -1
        self.cl_idx = -1
        self.br_idx = -1
        self.i_idx = -1
        
        # Initialize elements array
        self.n_elements = len(element_bounds)
        self.elements = <Element*>malloc(self.n_elements * sizeof(Element))
        
        # Sort elements by mass (heaviest first for better pruning)
        sorted_elements = sorted(element_bounds.items(), 
                               key=lambda x: ATOMIC_MASSES.get(x[0], 0), reverse=True)
        
        cdef int i, j
        cdef bytes symbol_bytes
        
        for i, (symbol, (min_count, max_count)) in enumerate(sorted_elements):
            # Copy symbol (ensure null termination)
            symbol_bytes = symbol.encode('ascii')
            for j in range(min(3, len(symbol_bytes))):
                self.elements[i].symbol[j] = symbol_bytes[j]
            self.elements[i].symbol[min(3, len(symbol_bytes))] = 0
            
            self.elements[i].mass = ATOMIC_MASSES.get(symbol, 0.0)
            self.elements[i].min_count = min_count
            self.elements[i].max_count = max_count
            
            # Store indices for constraint elements
            if symbol == 'C':
                self.c_idx = i
            elif symbol == 'H':
                self.h_idx = i
            elif symbol == 'N':
                self.n_idx = i
            elif symbol == 'P':
                self.p_idx = i
            elif symbol == 'F':
                self.f_idx = i
            elif symbol == 'Cl':
                self.cl_idx = i
            elif symbol == 'Br':
                self.br_idx = i
            elif symbol == 'I':
                self.i_idx = i
        
        # Initialize residue tables
        self._initialize_residue_tables()
    
    def __dealloc__(self):
        """Clean up allocated memory."""
        if self.elements:
            free(self.elements)
        if self.min_residues:
            free(self.min_residues)
        if self.max_residues:
            free(self.max_residues)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _initialize_residue_tables(self):
        """
        Initialize residue tables for efficient pruning.
        """
        cdef int i
        cdef double min_mass, max_mass
        
        # Allocate memory for residue tables
        self.min_residues = <double*>malloc(self.n_elements * sizeof(double))
        self.max_residues = <double*>malloc(self.n_elements * sizeof(double))
        
        # Compute cumulative minimum and maximum masses from each position
        min_mass = 0.0
        max_mass = 0.0
        
        for i in range(self.n_elements - 1, -1, -1):
            min_mass += self.elements[i].min_count * self.elements[i].mass
            max_mass += self.elements[i].max_count * self.elements[i].mass
            self.min_residues[i] = min_mass
            self.max_residues[i] = max_mass
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _check_chemical_constraints(self, int* formula) nogil:
        """
        Check if the current formula satisfies chemical constraints.
        Returns True if constraints are satisfied, False otherwise.
        """
        cdef int c_count, h_count, n_count, p_count, x_count, hetero_count
        cdef double dbe
        cdef int i
        
        # Skip constraint checking if not enabled
        if not self.use_dbe_constraint and not self.use_hetero_constraint:
            return True
        
        # Get element counts
        c_count = formula[self.c_idx] if self.c_idx >= 0 else 0
        h_count = formula[self.h_idx] if self.h_idx >= 0 else 0
        n_count = formula[self.n_idx] if self.n_idx >= 0 else 0
        p_count = formula[self.p_idx] if self.p_idx >= 0 else 0
        
        # Skip if no carbon (most constraints require carbon)
        if c_count == 0 and (self.use_dbe_constraint or self.use_hetero_constraint):
            return False
        
        # Calculate halogen count (F, Cl, Br, I)
        x_count = 0
        if self.f_idx >= 0:
            x_count += formula[self.f_idx]
        if self.cl_idx >= 0:
            x_count += formula[self.cl_idx]
        if self.br_idx >= 0:
            x_count += formula[self.br_idx]
        if self.i_idx >= 0:
            x_count += formula[self.i_idx]
        
        # Check DBE constraint
        if self.use_dbe_constraint and c_count > 0:
            dbe = c_count + 1 - (h_count - x_count + n_count + p_count) / 2.0
            if dbe < self.min_dbe or dbe > self.max_dbe:
                return False
        
        # Check heteroatom ratio constraint
        if self.use_hetero_constraint and c_count > 0:
            hetero_count = 0
            for i in range(self.n_elements):
                # Skip C and H when counting heteroatoms
                if i != self.c_idx and i != self.h_idx:
                    hetero_count += formula[i]
            
            if <double>hetero_count / <double>c_count > self.max_hetero_ratio:
                return False
        
        return True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _should_check_partial_constraints(self, int level) nogil:
        """
        Determine if we should check partial constraints at this level.
        We check when we have processed key elements (C, H, N, P, halogens).
        """
        cdef int processed_key_elements = 0
        
        # Count how many key elements we've already processed
        if self.c_idx >= 0 and self.c_idx <= level:
            processed_key_elements += 1
        if self.h_idx >= 0 and self.h_idx <= level:
            processed_key_elements += 1
        if self.n_idx >= 0 and self.n_idx <= level:
            processed_key_elements += 1
        if self.p_idx >= 0 and self.p_idx <= level:
            processed_key_elements += 1
        
        # Check partial constraints if we have at least 2 key elements
        return processed_key_elements >= 2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _check_partial_constraints(self, int* formula, int level) nogil:
        """
        Check partial constraints during enumeration to prune search space early.
        This is a simplified version that can work with incomplete formulas.
        """
        cdef int c_count, h_count, n_count, p_count, x_count, hetero_count
        cdef double dbe, min_possible_dbe, max_possible_dbe
        cdef int i, remaining_h, remaining_n, remaining_p, remaining_x
        
        # Skip if constraints not enabled
        if not self.use_dbe_constraint and not self.use_hetero_constraint:
            return True
        
        # Get current element counts (only for processed elements)
        c_count = formula[self.c_idx] if (self.c_idx >= 0 and self.c_idx < level) else 0
        h_count = formula[self.h_idx] if (self.h_idx >= 0 and self.h_idx < level) else 0
        n_count = formula[self.n_idx] if (self.n_idx >= 0 and self.n_idx < level) else 0
        p_count = formula[self.p_idx] if (self.p_idx >= 0 and self.p_idx < level) else 0
        
        # Calculate current halogen count
        x_count = 0
        if self.f_idx >= 0 and self.f_idx < level:
            x_count += formula[self.f_idx]
        if self.cl_idx >= 0 and self.cl_idx < level:
            x_count += formula[self.cl_idx]
        if self.br_idx >= 0 and self.br_idx < level:
            x_count += formula[self.br_idx]
        if self.i_idx >= 0 and self.i_idx < level:
            x_count += formula[self.i_idx]
        
        # Skip if no carbon processed yet
        if c_count == 0:
            return True
        
        # Check partial DBE constraint with bounds for remaining elements
        if self.use_dbe_constraint:
            # Get remaining possible counts for unprocessed elements
            remaining_h = self.elements[self.h_idx].max_count if (self.h_idx >= level and self.h_idx >= 0) else 0
            remaining_n = self.elements[self.n_idx].max_count if (self.n_idx >= level and self.n_idx >= 0) else 0
            remaining_p = self.elements[self.p_idx].max_count if (self.p_idx >= level and self.p_idx >= 0) else 0
            
            remaining_x = 0
            if self.f_idx >= level and self.f_idx >= 0:
                remaining_x += self.elements[self.f_idx].max_count
            if self.cl_idx >= level and self.cl_idx >= 0:
                remaining_x += self.elements[self.cl_idx].max_count
            if self.br_idx >= level and self.br_idx >= 0:
                remaining_x += self.elements[self.br_idx].max_count
            if self.i_idx >= level and self.i_idx >= 0:
                remaining_x += self.elements[self.i_idx].max_count
            
            # Calculate DBE bounds
            # Minimum DBE: maximize H, N, P, X contributions (most negative)
            min_possible_dbe = c_count + 1 - ((h_count + remaining_h) - (x_count + remaining_x) + (n_count + remaining_n) + (p_count + remaining_p)) / 2.0
            
            # Maximum DBE: minimize H, N, P, X contributions (most positive)
            max_possible_dbe = c_count + 1 - (h_count - x_count + n_count + p_count) / 2.0
            
            # Check if DBE bounds can satisfy constraints
            if max_possible_dbe < self.min_dbe or min_possible_dbe > self.max_dbe:
                return False
        
        # Check partial heteroatom ratio (this is more approximate)
        if self.use_hetero_constraint:
            hetero_count = 0
            for i in range(level):
                # Skip C and H when counting heteroatoms
                if i != self.c_idx and i != self.h_idx:
                    hetero_count += formula[i]
            
            # Early rejection if already too many heteroatoms
            if <double>hetero_count / <double>c_count > self.max_hetero_ratio:
                return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _can_reach_target(self, double current_mass, int level) nogil:
        """
        Check if target mass can still be reached from current position.
        """
        cdef double remaining_min, remaining_max
        
        if level >= self.n_elements:
            return fabs(current_mass - self.target_mass) <= self.tolerance
        
        remaining_min = current_mass + self.min_residues[level]
        remaining_max = current_mass + self.max_residues[level]
        
        return (self.target_mass - self.tolerance <= remaining_max and 
                remaining_min <= self.target_mass + self.tolerance)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _decompose_recursive(self, int* formula, double current_mass, int level):
        """
        Recursive decomposition with aggressive pruning and chemical constraints.
        """
        cdef int count, min_count, max_count
        cdef double new_mass, element_mass
        cdef double mass_diff
        cdef int max_possible_count
        
        # Base case: all elements processed
        if level >= self.n_elements:
            mass_diff = fabs(current_mass - self.target_mass)
            if mass_diff <= self.tolerance:
                # Check chemical constraints before adding result
                if self._check_chemical_constraints(formula):
                    # Found valid formula
                    self._add_result(formula)
            return
        
        # Early termination if too many results
        if len(self.results) >= self.max_results:
            return
        
        # Pruning: check if target can still be reached
        if not self._can_reach_target(current_mass, level):
            return
        
        # Get element properties
        element_mass = self.elements[level].mass
        min_count = self.elements[level].min_count
        max_count = self.elements[level].max_count
        
        # Additional pruning: adjust max count based on remaining mass
        if element_mass > 0:
            max_possible_count = <int>((self.target_mass + self.tolerance - current_mass) / element_mass)
            max_count = min(max_count, max_possible_count)
        
        # Ensure we don't go below minimum
        if max_count < min_count:
            return
        
        # Try all possible counts for this element
        for count in range(min_count, max_count + 1):
            new_mass = current_mass + count * element_mass
            
            # Pruning: skip if mass already too high
            if new_mass > self.target_mass + self.tolerance:
                break
            
            formula[level] = count
            
            # Early constraint checking for critical elements to prune search space
            # Only check constraints if we have enough elements processed
            if level >= self.n_elements - 1 or self._should_check_partial_constraints(level):
                if not self._check_partial_constraints(formula, level + 1):
                    continue
            
            self._decompose_recursive(formula, new_mass, level + 1)
    
    cdef void _add_result(self, int* formula):
        """Add a valid formula to results."""
        cdef int i, j
        cdef bytes symbol_bytes
        
        result = {}
        
        for i in range(self.n_elements):
            if formula[i] > 0:
                # Safely extract symbol from C char array
                symbol_bytes = self.elements[i].symbol
                try:
                    symbol = symbol_bytes.decode('ascii').rstrip('\x00')
                    if symbol:  # Only add non-empty symbols
                        result[symbol] = formula[i]
                except UnicodeDecodeError:
                    # Fallback: construct symbol byte by byte
                    symbol_chars = []
                    for j in range(4):  # max 3 chars + null terminator
                        if self.elements[i].symbol[j] == 0:
                            break
                        if 32 <= self.elements[i].symbol[j] <= 126:  # printable ASCII
                            symbol_chars.append(chr(self.elements[i].symbol[j]))
                    symbol = ''.join(symbol_chars)
                    if symbol:
                        result[symbol] = formula[i]
        
        self.results.append(result)
    
    def decompose(self) -> List[Dict[str, int]]:
        """
        Perform mass decomposition to find all valid molecular formulas.
        """
        cdef int* formula = <int*>malloc(self.n_elements * sizeof(int))
        cdef int i
        
        # Initialize formula array
        for i in range(self.n_elements):
            formula[i] = 0
        
        try:
            # Clear previous results
            self.results = []
            
            # Start recursive decomposition
            self._decompose_recursive(formula, 0.0, 0)
            
            return self.results.copy()
        
        finally:
            free(formula)

def cython_decompose_mass(target_mass: float, 
                          element_bounds: Dict[str, Tuple[int, int]],
                          tolerance_ppm: float = 5.0,
                          max_results: int = 1000000,
                          min_dbe: Optional[float] = None,
                          max_dbe: Optional[float] = None,
                          max_hetero_ratio: Optional[float] = None) -> List[Dict[str, int]]:
    """
    High-level function for ultra-fast mass decomposition using Cython with chemical constraints.
    
    Args:
        target_mass: Target mass to decompose
        element_bounds: Dictionary mapping element symbols to (min_count, max_count)
        tolerance_ppm: Mass tolerance in parts per million
        max_results: Maximum number of results to return
        min_dbe: Minimum double bond equivalents (None = no constraint)
        max_dbe: Maximum double bond equivalents (None = no constraint)
        max_hetero_ratio: Maximum ratio of heteroatoms to carbons (None = no constraint)
    
    Returns:
        List of molecular formulas as dictionaries
    """
    # Set default values for constraints
    cdef double c_min_dbe = min_dbe if min_dbe is not None else -1000.0
    cdef double c_max_dbe = max_dbe if max_dbe is not None else 1000.0
    cdef double c_max_hetero_ratio = max_hetero_ratio if max_hetero_ratio is not None else 1000.0
    
    decomposer = CythonSiriusDecomposer(element_bounds, target_mass, tolerance_ppm, max_results,
                                       c_min_dbe, c_max_dbe, c_max_hetero_ratio)
    return decomposer.decompose()
