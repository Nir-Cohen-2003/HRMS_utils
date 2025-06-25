# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython implementation of the SIRIUS mass decomposition algorithm based on the
original Java implementation's "Money Changing Problem" approach.
"""
cimport cython
from libc.stdlib cimport malloc, free, calloc
from libc.math cimport ceil, floor, fmax, fabs, round as c_round
from typing import List, Dict, Tuple

# Standard atomic masses
ATOMIC_MASSES = {
    'C': 12.0000000, 'H': 1.0078250, 'O': 15.9949146, 'N': 14.0030740,
    'P': 30.9737620, 'S': 31.9720718, 'F': 18.9984032, 'Cl': 34.9688527,
    'Br': 78.9183376, 'I': 126.9044719, 'Si': 27.9769271, 'Na': 22.9897693,
    'K': 38.9637069, 'Ca': 39.9625912, 'Mg': 23.9850423, 'Fe': 55.9349421,
    'Zn': 63.9291466, 'Se': 79.9165218, 'B': 11.0093054, 'Al': 26.9815386
}

cdef long long LLONG_MAX = 9223372036854775807

cdef struct Weight:
    char symbol[4]
    double mass
    long long integer_mass
    int min_count
    int max_count

cdef bint _check_dbe(dict formula, double min_dbe, double max_dbe):
    """
    Checks if a formula satisfies Double Bond Equivalent (DBE) constraints.
    Uses the same formula as sirius_decomposer.pyx for consistency.
    """
    cdef int c_count = formula.get('C', 0)
    cdef int h_count = formula.get('H', 0)
    cdef int n_count = formula.get('N', 0)
    cdef int p_count = formula.get('P', 0)
    cdef int x_count = (formula.get('F', 0) + formula.get('Cl', 0) + 
                       formula.get('Br', 0) + formula.get('I', 0))

    if c_count == 0:
        return True  # Cannot calculate DBE without Carbon

    # DBE formula consistent with the other Cython implementation
    dbe = c_count + 1.0 - (h_count - x_count + n_count) / 2.0 - p_count
    
    # Check if DBE is within range and is close to an integer
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

cdef class CythonJavaStyleDecomposer:
    cdef long long** ERT
    cdef double precision
    cdef Weight* weights
    cdef int n_elements
    cdef double min_error, max_error
    cdef list element_symbols
    cdef bint is_initialized

    def __cinit__(self, element_bounds: Dict[str, Tuple[int, int]]):
        self.is_initialized = False
        self.ERT = NULL
        self.weights = NULL
        
        # A fixed precision, similar to the Java implementation's comment
        self.precision = 1.0 / 5963.337687 

        self.n_elements = len(element_bounds)
        self.weights = <Weight*>malloc(self.n_elements * sizeof(Weight))
        if not self.weights:
            raise MemoryError()

        # Sort elements by mass (smallest first, as in Java impl)
        sorted_elements = sorted(element_bounds.items(), 
                               key=lambda x: ATOMIC_MASSES.get(x[0], 0.0))
        
        self.element_symbols = []
        cdef int i, j
        cdef bytes symbol_bytes
        for i, (symbol, (min_count, max_count)) in enumerate(sorted_elements):
            self.element_symbols.append(symbol)
            symbol_bytes = symbol.encode('ascii')
            for j in range(min(3, len(symbol_bytes))):
                self.weights[i].symbol[j] = symbol_bytes[j]
            self.weights[i].symbol[min(3, len(symbol_bytes))] = 0
            self.weights[i].mass = ATOMIC_MASSES.get(symbol, 0.0)
            self.weights[i].min_count = min_count
            self.weights[i].max_count = max_count

    def __dealloc__(self):
        cdef int i
        if self.ERT is not NULL:
            # Check if weights is valid before accessing it
            if self.weights is not NULL and self.weights[0].integer_mass > 0:
                for i in range(self.weights[0].integer_mass):
                    if self.ERT[i] is not NULL:
                        free(self.ERT[i])
            free(self.ERT)
        if self.weights is not NULL:
            free(self.weights)

    cdef void _init(self):
        if self.is_initialized:
            return
        self._discretize_masses()
        self._divide_by_gcd()
        self._calc_ert()
        self._compute_errors()
        self.is_initialized = True

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
        if not self.ERT: raise MemoryError()
        
        cdef int i, j, p, r, argmin
        for i in range(first_long_val):
            self.ERT[i] = <long long*>malloc(self.n_elements * sizeof(long long))
            if not self.ERT[i]: raise MemoryError()

        self.ERT[0][0] = 0
        for i in range(1, first_long_val):
            self.ERT[i][0] = LLONG_MAX

        cdef long long d, n
        for j in range(1, self.n_elements):
            self.ERT[0][j] = 0
            d = c_gcd(first_long_val, self.weights[j].integer_mass)
            for p in range(d):
                n = LLONG_MAX
                argmin = p
                for i in range(p, first_long_val, d):
                    if self.ERT[i][j-1] < n:
                        n = self.ERT[i][j-1]
                        argmin = i
                
                if n == LLONG_MAX:
                    for i in range(p, first_long_val, d):
                        self.ERT[i][j] = LLONG_MAX
                else:
                    # This loop is a direct translation of the Java implementation's logic
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
            if self.weights[i].mass == 0: continue
            error = (self.precision * self.weights[i].integer_mass - self.weights[i].mass) / self.weights[i].mass
            if error < self.min_error: self.min_error = error
            if error > self.max_error: self.max_error = error

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
        if m < 0: return False
        return self.ERT[m % a1][i] <= m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _integer_decompose(self, long long mass, int* bounds):
        cdef list results = []
        cdef int k = self.n_elements - 1
        if k < 0: return []
        cdef long long a = self.weights[0].integer_mass
        if a <= 0: return []

        cdef int* c = <int*>calloc(k + 1, sizeof(int))
        if not c: raise MemoryError()
        
        cdef int i = k
        cdef int j
        cdef long long m = mass
        
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

    def decompose(self, double mass, double tolerance_ppm, double min_dbe=-100.0, double max_dbe=100.0):
        self._init()
        
        cdef double tolerance_mass = mass * tolerance_ppm / 1e6
        cdef double mass_from = mass - tolerance_mass
        cdef double mass_to = mass + tolerance_mass
        cdef int i
        cdef double min_mass_offset = 0.0
        cdef int* min_counts = <int*>calloc(self.n_elements, sizeof(int))
        cdef int* max_count_ranges = <int*>calloc(self.n_elements, sizeof(int))
        cdef double c_mass_from, c_mass_to
        cdef long long int_mass_from, int_mass_to
        cdef list all_results
        cdef long long m
        cdef list raw_decompositions
        cdef dict raw_decomp, final_decomp
        cdef double real_mass
        cdef bint bounds_ok

        if mass_to < 0: return []

        if not min_counts or not max_count_ranges: raise MemoryError()

        try:
            for i in range(self.n_elements):
                min_counts[i] = self.weights[i].min_count
                max_count_ranges[i] = self.weights[i].max_count - self.weights[i].min_count
                if min_counts[i] > 0:
                    min_mass_offset += min_counts[i] * self.weights[i].mass

            c_mass_from = mass_from - min_mass_offset
            c_mass_to = mass_to - min_mass_offset

            int_mass_from, int_mass_to = self._integer_bound(c_mass_from, c_mass_to)
            
            all_results = []
            for m in range(int_mass_from, int_mass_to + 1):
                if m < 0: continue
                raw_decompositions = self._integer_decompose(m, max_count_ranges)
                
                for raw_decomp in raw_decompositions:
                    final_decomp = {}
                    
                    # Construct final formula by adding min_counts
                    for i in range(self.n_elements):
                        symbol_str = self.element_symbols[i]
                        base_count = min_counts[i]
                        additional_count = raw_decomp.get(symbol_str, 0)
                        total_count = base_count + additional_count
                        
                        if total_count > 0:
                            final_decomp[symbol_str] = total_count
                    
                    # Check mass and DBE constraints
                    real_mass = 0.0
                    for symbol, count in final_decomp.items():
                        for i in range(self.n_elements):
                            if self.element_symbols[i] == symbol:
                                real_mass += count * self.weights[i].mass
                                break

                    if mass_from <= real_mass <= mass_to:
                        if _check_dbe(final_decomp, min_dbe, max_dbe):
                            if final_decomp not in all_results:
                                all_results.append(final_decomp)
        finally:
            free(min_counts)
            free(max_count_ranges)
        
        return all_results
