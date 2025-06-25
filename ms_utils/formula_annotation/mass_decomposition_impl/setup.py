"""
Setup script for compiling the unified Cython mass decomposition algorithm with OpenMP support.

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

# Define the extensions
extensions = [
    Extension(
        "mass_decomposer",
        ["mass_decomposer.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=openmp_compile_args,
        extra_link_args=openmp_link_args
    )
]

setup(
    name="mass_decomposer_suite",
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
