# cython: language_level=3, boundscheck=False, wraparound=False
"""
High-performance Cython implementation of the SIRIUS mass decomposition algorithm.
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
    Ultra-fast Cython implementation of SIRIUS mass decomposition algorithm.
    """
    
    cdef Element* elements
    cdef int n_elements
    cdef double* min_residues
    cdef double* max_residues
    cdef double target_mass
    cdef double tolerance
    cdef int max_results
    cdef list results
    
    def __cinit__(self, element_bounds: Dict[str, Tuple[int, int]], 
                  double target_mass, double tolerance_ppm = 5.0, 
                  int max_results = 1000000):
        """
        Initialize the mass decomposer.
        """
        self.target_mass = target_mass
        self.tolerance = target_mass * tolerance_ppm / 1e6
        self.max_results = max_results
        self.results = []
        
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
        Recursive decomposition with aggressive pruning.
        """
        cdef int count, min_count, max_count
        cdef double new_mass, element_mass
        cdef double mass_diff
        cdef int max_possible_count
        
        # Base case: all elements processed
        if level >= self.n_elements:
            mass_diff = fabs(current_mass - self.target_mass)
            if mass_diff <= self.tolerance:
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
                          max_results: int = 1000000) -> List[Dict[str, int]]:
    """
    High-level function for ultra-fast mass decomposition using Cython.
    """
    decomposer = CythonSiriusDecomposer(element_bounds, target_mass, tolerance_ppm, max_results)
    return decomposer.decompose()
