# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "cpp_filter.hpp" nogil:
    cdef cppclass PrecomputedMol:
        pass # No need to define members, just need the type
    
    vector[double] calculate_symmetric_distance_matrix(const vector[string]& smiles_list)
    double calculate_total_cost_symmetric(const vector[string]& smiles_list)

def calculate_distances_symmetric(smiles_list):
    """
    Calculates a symmetric distance matrix for a list of SMILES strings.

    This function passes the SMILES list to a C++ backend which handles
    parallel pre-computation and pairwise distance calculation.

    Args:
        smiles_list (list or iterable of str): A list of SMILES strings.

    Returns:
        numpy.ndarray: A 2D numpy array representing the symmetric distance matrix.
    """
    cdef Py_ssize_t n = len(smiles_list)
    if n == 0:
        return np.array([])

    # Convert Python strings to C++ strings
    cdef vector[string] cpp_smiles
    cpp_smiles.reserve(n)
    for s in smiles_list:
        cpp_smiles.push_back(s.encode('utf-8'))

    cdef vector[double] cpp_results
    # All parallel computation is inside the C++ layer.
    # We release the GIL for the duration of the C++ call.
    with nogil:
        cpp_results = calculate_symmetric_distance_matrix(cpp_smiles)

    # Check for empty result to avoid accessing invalid memory
    if cpp_results.empty():
        return np.array([]).reshape((n,n))
    
    # Copy the data to a NumPy array (not zero-copy, but safe)
    cdef Py_ssize_t size = cpp_results.size()
    py_results = np.empty(size, dtype=np.float64)
    cdef double[::1] py_view = py_results
    
    # Copy data from C++ vector to NumPy array
    cdef Py_ssize_t i
    for i in range(size):
        py_view[i] = cpp_results[i]
    
    # Reshape the flat array into a 2D matrix
    return py_results.reshape((n, n))