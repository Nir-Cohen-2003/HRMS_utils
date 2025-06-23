"""
Setup script for compiling the Cython SIRIUS mass decomposition algorithm.

To compile:
    python setup.py build_ext --inplace

To install:
    pip install -e .
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extensions
extensions = [
    Extension(
        "sirius_decomposer",
        ["sirius_decomposer.pyx"],
        include_dirs=[numpy.get_include()],
        language_level=3,
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
            'embedsignature': True
        }
    )
]

setup(
    name="mass_decomposer_suite",
    ext_modules=cythonize(extensions, 
                         compiler_directives={'language_level': 3}),
    zip_safe=False,
    install_requires=[
        "numpy",
        "cython"
    ]
)
