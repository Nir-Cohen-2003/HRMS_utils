# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Unified Cython implementation of mass decomposition algorithms with OpenMP parallelization.
Supports both recursive and money-changing strategies.
"""
cimport cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free, calloc
from libc.math cimport ceil, floor, fmax, fabs, round as c_round
from typing import List, Dict, Tuple
import numpy as np
cimport numpy as cnp

# Standard atomic masses
ATOMIC_MASSES = {
    'C': 12.0000000, 'H': 1.0078250, 'O': 15.9949146, 'N': 14.0030740,
    'P': 30.9737620, 'S': 31.9720718, 'F': 18.9984032, 'Cl': 34.9688527,
    'Br': 78.9183376, 'I': 126.9044719, 'Si': 27.9769271, 'Na': 22.9897693,
    'K': 38.9637069, 'Ca': 39.9625912, 'Mg': 23.9850423, 'Fe': 55.9349421,
    'Zn': 63.9291466, 'Se': 79.9165218, 'B': 11.0093054, 'Al': 26.9815386
}

cdef long long LLONG_MAX = 9223372036854775807

cdef struct Element:
    char symbol[4]
    double mass
    int min_count
    int max_count

cdef struct Weight:
    char symbol[4]
    double mass
    long long integer_mass
    int min_count
    int max_count

cdef bint _check_dbe(dict formula, double min_dbe, double max_dbe):
    """Check if a formula satisfies Double Bond Equivalent (DBE) constraints."""
    cdef int c_count = formula.get('C', 0)
    cdef int h_count = formula.get('H', 0)
    cdef int n_count = formula.get('N', 0)
    cdef int p_count = formula.get('P', 0)
    cdef int x_count = (formula.get('F', 0) + formula.get('Cl', 0) + 
                       formula.get('Br', 0) + formula.get('I', 0))

    if c_count == 0:
        return False

    dbe = (2.0 + 2.0*c_count + 3.0*p_count + n_count - h_count - x_count) / 2.0
    
    if dbe < min_dbe or dbe > max_dbe:
        return False
    if fabs(dbe - c_round(dbe)) > 1e-8:
        return False
    
    return True

cdef long long c_gcd(long long u, long long v):
    cdef long long r
    while v != 0:
        r = u % v
        u = v
        v = r
    return u

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class UnifiedMassDecomposer:
    """Unified mass decomposer supporting both recursive and money-changing strategies."""
    
    # Common fields
    cdef Element* elements
    cdef int n_elements
    cdef double tolerance
    cdef int max_results
    cdef list element_symbols
    cdef double min_dbe, max_dbe
    cdef double max_hetero_ratio
    cdef bint use_dbe_constraint
    cdef bint use_hetero_constraint
    cdef str strategy
    cdef double target_mass
    cdef list results
    
    # Recursive strategy fields
    cdef double* min_residues
    cdef double* max_residues
    cdef int c_idx, h_idx, n_idx, p_idx, f_idx, cl_idx, br_idx, i_idx
    
    # Money-changing strategy fields
    cdef long long** ERT
    cdef double precision
    cdef Weight* weights
    cdef double min_error, max_error
    cdef bint is_initialized

    def __cinit__(self, element_bounds: Dict[str, Tuple[int, int]], str strategy="money_changing"):
        self.is_initialized = False
        self.ERT = NULL
        self.weights = NULL
        self.elements = NULL
        self.min_residues = NULL
        self.max_residues = NULL
        self.results = []
        self.strategy = strategy
        
        self.n_elements = len(element_bounds)
        self.element_symbols = []
        
        # Initialize common elements array
        self.elements = <Element*>malloc(self.n_elements * sizeof(Element))
        if not self.elements:
            raise MemoryError()
        
        # Default constraints
        self.min_dbe = 0.0
        self.max_dbe = 40.0
        self.max_hetero_ratio = 1000.0  # Effectively disabled by default
        self.use_dbe_constraint = True
        self.use_hetero_constraint = False  # Disabled by default
        self.max_results = 10000
        
        if strategy == "money_changing":
            self._init_money_changing(element_bounds)
        else:  # recursive
            self._init_recursive(element_bounds)

    def __dealloc__(self):
        """Clean up memory."""
        cdef int i
        cdef long long first_val
        
        if self.elements:
            free(self.elements)
        if self.min_residues:
            free(self.min_residues)
        if self.max_residues:
            free(self.max_residues)
        if self.weights:
            free(self.weights)
        if self.ERT:
            first_val = self.weights[0].integer_mass if self.weights else 0
            for i in range(first_val):
                if self.ERT[i]:
                    free(self.ERT[i])
            free(self.ERT)

    cdef void _init_recursive(self, element_bounds):
        """Initialize for recursive strategy."""
        # Sort elements by mass (heaviest first for better pruning)
        sorted_elements = sorted(element_bounds.items(), 
                               key=lambda x: ATOMIC_MASSES.get(x[0], 0), reverse=True)
        
        # Initialize element indices
        self.c_idx = -1
        self.h_idx = -1
        self.n_idx = -1
        self.p_idx = -1
        self.f_idx = -1
        self.cl_idx = -1
        self.br_idx = -1
        self.i_idx = -1
        
        cdef int i, j
        cdef bytes symbol_bytes
        
        for i, (symbol, (min_count, max_count)) in enumerate(sorted_elements):
            self.element_symbols.append(symbol)
            symbol_bytes = symbol.encode('ascii')
            for j in range(min(3, len(symbol_bytes))):
                self.elements[i].symbol[j] = symbol_bytes[j]
            self.elements[i].symbol[min(3, len(symbol_bytes))] = 0
            
            self.elements[i].mass = ATOMIC_MASSES.get(symbol, 0.0)
            self.elements[i].min_count = min_count
            self.elements[i].max_count = max_count
            
            # Store indices for constraint elements
            if symbol == 'C': self.c_idx = i
            elif symbol == 'H': self.h_idx = i
            elif symbol == 'N': self.n_idx = i
            elif symbol == 'P': self.p_idx = i
            elif symbol == 'F': self.f_idx = i
            elif symbol == 'Cl': self.cl_idx = i
            elif symbol == 'Br': self.br_idx = i
            elif symbol == 'I': self.i_idx = i

    cdef void _init_money_changing(self, element_bounds):
        """Initialize for money-changing strategy."""
        self.precision = 1.0 / 5963.337687
        
        self.weights = <Weight*>malloc(self.n_elements * sizeof(Weight))
        if not self.weights:
            raise MemoryError()

        # Sort elements by mass (smallest first for money-changing)
        sorted_elements = sorted(element_bounds.items(), 
                               key=lambda x: ATOMIC_MASSES.get(x[0], 0.0))
        
        # Initialize element indices for constraints
        self.c_idx = -1
        self.h_idx = -1
        self.n_idx = -1
        self.p_idx = -1
        self.f_idx = -1
        self.cl_idx = -1
        self.br_idx = -1
        self.i_idx = -1
        
        cdef int i, j
        cdef bytes symbol_bytes
        for i, (symbol, (min_count, max_count)) in enumerate(sorted_elements):
            self.element_symbols.append(symbol)
            symbol_bytes = symbol.encode('ascii')
            for j in range(min(3, len(symbol_bytes))):
                self.weights[i].symbol[j] = symbol_bytes[j]
                self.elements[i].symbol[j] = symbol_bytes[j]
            self.weights[i].symbol[min(3, len(symbol_bytes))] = 0
            self.elements[i].symbol[min(3, len(symbol_bytes))] = 0
            
            self.weights[i].mass = ATOMIC_MASSES.get(symbol, 0.0)
            self.weights[i].min_count = min_count
            self.weights[i].max_count = max_count
            
            self.elements[i].mass = ATOMIC_MASSES.get(symbol, 0.0)
            self.elements[i].min_count = min_count
            self.elements[i].max_count = max_count
            
            # Store indices for constraint elements
            if symbol == 'C': self.c_idx = i
            elif symbol == 'H': self.h_idx = i
            elif symbol == 'N': self.n_idx = i
            elif symbol == 'P': self.p_idx = i
            elif symbol == 'F': self.f_idx = i
            elif symbol == 'Cl': self.cl_idx = i
            elif symbol == 'Br': self.br_idx = i
            elif symbol == 'I': self.i_idx = i

    def decompose(self, double target_mass, double tolerance_ppm=5.0, 
                  double min_dbe=0.0, double max_dbe=40.0, 
                  double max_hetero_ratio=1000.0, int max_results=10000):
        """
        Main decomposition method that uses the selected strategy.
        """
        self.target_mass = target_mass
        self.tolerance = target_mass * tolerance_ppm / 1e6
        self.min_dbe = min_dbe
        self.max_dbe = max_dbe
        self.max_hetero_ratio = max_hetero_ratio
        self.use_hetero_constraint = max_hetero_ratio < 100.0  # Only use if reasonable value
        self.max_results = max_results
        self.results = []
        
        if self.strategy == "money_changing":
            return self._decompose_money_changing()
        else:
            return self._decompose_recursive()

    def _decompose_recursive(self):
        """Recursive strategy decomposition."""
        self._initialize_residue_tables()
        
        # Allocate formula array
        cdef int* formula = <int*>calloc(self.n_elements, sizeof(int))
        if not formula:
            raise MemoryError()
        
        try:
            self._decompose_recursive_impl(formula, 0.0, 0)
        finally:
            free(formula)
        
        return self.results

    def _decompose_money_changing(self):
        """Money-changing strategy decomposition."""
        if not self.is_initialized:
            self._discretize_masses()
            self._divide_by_gcd()
            self._calc_ert()
            self._compute_errors()
            self.is_initialized = True
        
        # Convert mass bounds to integer bounds
        mass_from = self.target_mass - self.tolerance
        mass_to = self.target_mass + self.tolerance
        start, end = self._integer_bound(mass_from, mass_to)
        
        # Create bounds array for elements (use actual max bounds)
        cdef int* bounds = <int*>malloc(self.n_elements * sizeof(int))
        if not bounds:
            raise MemoryError()
        
        cdef int i
        for i in range(self.n_elements):
            bounds[i] = self.weights[i].max_count
        
        try:
            results = []
            for mass in range(start, end + 1):
                mass_results = self._integer_decompose(mass, bounds)
                for result in mass_results:
                    if _check_dbe(result, self.min_dbe, self.max_dbe):
                        results.append(result)
                        if len(results) >= self.max_results:
                            break
                if len(results) >= self.max_results:
                    break
            return results
        finally:
            free(bounds)

    cdef void _discretize_masses(self):
        cdef int i
        for i in range(self.n_elements):
            self.weights[i].integer_mass = <long long>(self.weights[i].mass / self.precision)

    cdef void _divide_by_gcd(self):
        if self.n_elements < 2:
            return
            
        cdef long long d = c_gcd(self.weights[0].integer_mass, self.weights[1].integer_mass)
        cdef int i
        for i in range(2, self.n_elements):
            d = c_gcd(d, self.weights[i].integer_mass)
            if d == 1:
                break
        
        if d > 1:
            self.precision *= d
            for i in range(self.n_elements):
                self.weights[i].integer_mass = self.weights[i].integer_mass // d

    cdef void _calc_ert(self):
        cdef long long first_long_val = self.weights[0].integer_mass
        if first_long_val <= 0:
            raise ValueError("First element mass is zero or negative after discretization.")

        self.ERT = <long long**>malloc(first_long_val * sizeof(long long*))
        if not self.ERT: 
            raise MemoryError()
        
        cdef int i, j, p, r
        for i in range(first_long_val):
            self.ERT[i] = <long long*>malloc(self.n_elements * sizeof(long long))
            if not self.ERT[i]: 
                raise MemoryError()

        self.ERT[0][0] = 0
        for i in range(1, first_long_val):
            self.ERT[i][0] = LLONG_MAX

        cdef long long d, n
        for j in range(1, self.n_elements):
            self.ERT[0][j] = 0
            d = c_gcd(first_long_val, self.weights[j].integer_mass)
            for p in range(d):
                n = LLONG_MAX
                for i in range(p, first_long_val, d):
                    if self.ERT[i][j-1] < n:
                        n = self.ERT[i][j-1]
                
                if n == LLONG_MAX:
                    for i in range(p, first_long_val, d):
                        self.ERT[i][j] = LLONG_MAX
                else:
                    for i in range(first_long_val // d):
                        n += self.weights[j].integer_mass
                        r = <int>(n % first_long_val)
                        if self.ERT[r][j-1] < n:
                            n = self.ERT[r][j-1]
                        self.ERT[r][j] = n

    cdef void _compute_errors(self):
        self.min_error = 0.0
        self.max_error = 0.0
        cdef int i
        cdef double error
        for i in range(self.n_elements):
            if self.weights[i].mass == 0: 
                continue
            error = (self.precision * self.weights[i].integer_mass - self.weights[i].mass) / self.weights[i].mass
            if error < self.min_error: 
                self.min_error = error
            if error > self.max_error: 
                self.max_error = error

    cdef tuple _integer_bound(self, double mass_from, double mass_to):
        cdef double from_d = ceil((1 + self.min_error) * mass_from / self.precision)
        cdef double to_d = floor((1 + self.max_error) * mass_to / self.precision)
        
        if from_d > LLONG_MAX or to_d > LLONG_MAX:
            raise ValueError("Mass too large for 64-bit integer space.")
            
        cdef long long start = <long long>fmax(0, from_d)
        cdef long long end = <long long>fmax(start, to_d)
        return start, end

    @cython.nogil
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _decomposable(self, int i, long long m, long long a1) nogil:
        if m < 0: 
            return False
        return self.ERT[m % a1][i] <= m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _integer_decompose(self, long long mass, int* bounds):
        cdef list results = []
        cdef int k = self.n_elements - 1
        if k < 0: 
            return []
        cdef long long a = self.weights[0].integer_mass
        if a <= 0: 
            return []

        cdef int* c = <int*>calloc(k + 1, sizeof(int))
        if not c: 
            raise MemoryError()
        
        cdef int i = k
        cdef int j
        cdef long long m = mass
        cdef double actual_mass, mass_error_ppm
        
        try:
            while i <= k:
                if not self._decomposable(i, m, a):
                    while i <= k and not self._decomposable(i, m, a):
                        m += c[i] * self.weights[i].integer_mass
                        c[i] = 0
                        i += 1
                    
                    if i <= k:
                        m -= self.weights[i].integer_mass
                        c[i] += 1
                else:
                    while i > 0 and self._decomposable(i - 1, m, a):
                        i -= 1
                    
                    if i == 0:
                        if a > 0:
                            c[0] = <int>(m / a)
                        else:
                            c[0] = 0

                        # Check element bounds and mass tolerance
                        valid_formula = True
                        for j in range(k + 1):
                            if c[j] < self.weights[j].min_count or c[j] > self.weights[j].max_count:
                                valid_formula = False
                                break
                        
                        if valid_formula:
                            # Calculate actual mass and check tolerance
                            actual_mass = 0.0
                            for j in range(k + 1):
                                actual_mass += c[j] * self.weights[j].mass
                            
                            mass_error_ppm = abs(actual_mass - self.target_mass) / self.target_mass * 1e6
                            if mass_error_ppm <= self.tolerance * 1e6 / self.target_mass:
                                res = {}
                                for j in range(k + 1):
                                    if c[j] > 0:
                                        res[self.element_symbols[j]] = c[j]
                                if res:
                                    results.append(res)
                        i += 1
                    
                    while i <= k and c[i] >= bounds[i]:
                        m += c[i] * self.weights[i].integer_mass
                        c[i] = 0
                        i += 1

                    if i <= k:
                        m -= self.weights[i].integer_mass
                        c[i] += 1
        finally:
            free(c)
            
        return results

    cdef void _initialize_residue_tables(self):
        """Initialize residue tables for efficient pruning."""
        cdef int i
        cdef double min_mass, max_mass
        
        # Allocate memory for residue tables
        self.min_residues = <double*>malloc(self.n_elements * sizeof(double))
        self.max_residues = <double*>malloc(self.n_elements * sizeof(double))
        
        if not self.min_residues or not self.max_residues:
            raise MemoryError()
        
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
        """Check if the current formula satisfies chemical constraints."""
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
            dbe = (2.0 + 2.0*c_count + 3.0*p_count + n_count - h_count - x_count) / 2.0
            if dbe < self.min_dbe or dbe > self.max_dbe or fabs(dbe - c_round(dbe)) > 1e-8:
                return False

        # Check heteroatom ratio constraint
        if self.use_hetero_constraint and c_count > 0:
            hetero_count = 0
            for i in range(self.n_elements):
                if i != self.c_idx and i != self.h_idx:
                    hetero_count += formula[i]
            
            if <double>hetero_count / <double>c_count > self.max_hetero_ratio:
                return False
        
        return True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _can_reach_target(self, double current_mass, int level) nogil:
        """Check if target mass can still be reached from current position."""
        cdef double remaining_min, remaining_max
        
        if level >= self.n_elements:
            return fabs(current_mass - self.target_mass) <= self.tolerance
        
        remaining_min = current_mass + self.min_residues[level]
        remaining_max = current_mass + self.max_residues[level]
        
        return (self.target_mass - self.tolerance <= remaining_max and 
                remaining_min <= self.target_mass + self.tolerance)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _decompose_recursive_impl(self, int* formula, double current_mass, int level):
        """Recursive decomposition with aggressive pruning and chemical constraints."""
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
            self._decompose_recursive_impl(formula, new_mass, level + 1)

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


# Public API functions
def decompose_mass(double target_mass, dict element_bounds, 
                   str strategy="money_changing", double tolerance_ppm=5.0,
                   double min_dbe=0.0, double max_dbe=40.0, 
                   double max_hetero_ratio=1000.0, int max_results=10000):
    """
    Decompose a mass into possible molecular formulas.
    
    Args:
        target_mass: Target mass to decompose
        element_bounds: Dictionary of element bounds {element: (min, max)}
        strategy: "recursive" or "money_changing"
        tolerance_ppm: Mass tolerance in ppm
        min_dbe: Minimum double bond equivalents
        max_dbe: Maximum double bond equivalents
        max_hetero_ratio: Maximum heteroatom to carbon ratio
        max_results: Maximum number of results to return
        
    Returns:
        List of formula dictionaries
    """
    decomposer = UnifiedMassDecomposer(element_bounds, strategy)
    return decomposer.decompose(target_mass, tolerance_ppm, min_dbe, max_dbe, 
                               max_hetero_ratio, max_results)

# Add process pool support
from multiprocessing import Pool
import multiprocessing

cdef list _decompose_single_mass_helper(args):
    """Helper function for multiprocessing."""
    target_mass, element_bounds, strategy, tolerance_ppm, min_dbe, max_dbe, max_hetero_ratio, max_results = args
    decomposer = UnifiedMassDecomposer(element_bounds, strategy)
    return decomposer.decompose(target_mass, tolerance_ppm, min_dbe, max_dbe,
                               max_hetero_ratio, max_results)

@cython.boundscheck(False)
@cython.wraparound(False)
def decompose_mass_parallel(list target_masses, dict element_bounds,
                           str strategy="money_changing", double tolerance_ppm=5.0,
                           double min_dbe=0.0, double max_dbe=40.0,
                           double max_hetero_ratio=1000.0, int max_results=10000):
    """
    Decompose multiple masses in parallel using multiprocessing.
    
    Args:
        target_masses: List of target masses to decompose
        element_bounds: Dictionary of element bounds {element: (min, max)}
        strategy: "recursive" or "money_changing"
        tolerance_ppm: Mass tolerance in ppm
        min_dbe: Minimum double bond equivalents
        max_dbe: Maximum double bond equivalents
        max_hetero_ratio: Maximum heteroatom to carbon ratio
        max_results: Maximum number of results to return per mass
        
    Returns:
        List of lists of formula dictionaries
    """
    cdef int n_masses = len(target_masses)
    
    if n_masses == 0:
        return []
    
    if n_masses == 1:
        # Single mass - no need for parallelization overhead
        decomposer = UnifiedMassDecomposer(element_bounds, strategy)
        return [decomposer.decompose(target_masses[0], tolerance_ppm, min_dbe, max_dbe,
                                   max_hetero_ratio, max_results)]
    
    # Prepare arguments for parallel processing
    args_list = []
    for mass in target_masses:
        args_list.append((mass, element_bounds, strategy, tolerance_ppm, 
                         min_dbe, max_dbe, max_hetero_ratio, max_results))
    
    # Use multiprocessing for true parallelization
    max_workers = min(multiprocessing.cpu_count(), n_masses)
    
    try:
        with Pool(processes=max_workers) as pool:
            results = pool.map(_decompose_single_mass_helper, args_list)
        return results
    except Exception:
        # Fallback to sequential processing if multiprocessing fails
        results = []
        for args in args_list:
            results.append(_decompose_single_mass_helper(args))
        return results

@cython.boundscheck(False)
@cython.wraparound(False)
def decompose_spectrum_parallel(double precursor_mass, list fragment_masses,
                               dict element_bounds, str strategy="money_changing",
                               double tolerance_ppm=5.0, double min_dbe=0.0,
                               double max_dbe=40.0, double max_hetero_ratio=1000.0,
                               int max_results=10000):
    """
    Decompose a spectrum (precursor + fragments) in parallel using OpenMP.
    
    Args:
        precursor_mass: Precursor ion mass
        fragment_masses: List of fragment masses
        element_bounds: Dictionary of element bounds {element: (min, max)}
        strategy: "recursive" or "money_changing"
        tolerance_ppm: Mass tolerance in ppm
        min_dbe: Minimum double bond equivalents
        max_dbe: Maximum double bond equivalents
        max_hetero_ratio: Maximum heteroatom to carbon ratio
        max_results: Maximum number of results to return per mass
        
    Returns:
        Dictionary with 'precursor' and 'fragments' keys containing decomposition results
    """
    # Combine precursor and fragments into a single list for parallel processing
    all_masses = [precursor_mass] + fragment_masses
    
    # Decompose all masses in parallel
    all_results = decompose_mass_parallel(all_masses, element_bounds, strategy,
                                        tolerance_ppm, min_dbe, max_dbe, 
                                        max_hetero_ratio, max_results)
    
    # Split results back into precursor and fragments
    precursor_results = all_results[0]
    fragment_results = all_results[1:] if len(all_results) > 1 else []
    
    return {
        'precursor': precursor_results,
        'fragments': fragment_results
    }