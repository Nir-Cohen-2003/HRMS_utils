"""
Setup script for compiling the C++ mass decomposition algorithm with OpenMP support.

To compile:
    python setup.py build_ext --inplace

To install:
    pip install -e .
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# OpenMP flags
openmp_compile_args = []
openmp_link_args = []

# Detect compiler and set appropriate OpenMP flags
if os.name == 'nt':  # Windows
    openmp_compile_args = ['/openmp']
else:  # Unix-like systems
    openmp_compile_args = ['-fopenmp']
    openmp_link_args = ['-fopenmp']

# C++ compile flags
cpp_compile_args = ['-std=c++11', '-O3', '-march=native']
if os.name != 'nt':
    cpp_compile_args.extend(['-ffast-math', '-funroll-loops'])

# Define the extensions
extensions = [
    # C++ implementation with OpenMP
    Extension(
        "mass_decomposer_cpp",
        [
            "mass_decomposer_cpp.pyx", 
            "mass_decomposer_common.cpp",
            "mass_decomposer_money_changing.cpp",
            "mass_decomposer_parallel.cpp"
        ],
        include_dirs=[numpy.get_include(), "."],
        extra_compile_args=cpp_compile_args + openmp_compile_args,
        extra_link_args=openmp_link_args,
        language="c++"
    )
]

setup(
    name="mass_decomposer_cpp",
    ext_modules=cythonize(extensions, 
                         compiler_directives={
                             'language_level': 3,
                             'boundscheck': False,
                             'wraparound': False,
                             'initializedcheck': False,
                             'cdivision': True,
                             'embedsignature': True
                         }),
    zip_safe=False,
    install_requires=[
        "numpy",
        "cython"
    ]
)