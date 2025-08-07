from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os
import sys

# --- Compilation settings from the nested setup.py ---

# Define the base path for the extension module
ext_base_path = "src/ms_utils/formula_annotation/mass_decomposition_impl"

# OpenMP flags
openmp_compile_args = []
openmp_link_args = []

# C++ compile flags
cpp_compile_args = ['-std=c++11', '-O3']

# Detect compiler and set appropriate flags for the TARGET platform
# sys.platform is patched by conda-build to reflect the target platform
is_windows = sys.platform.startswith('win')

if is_windows:
    openmp_compile_args = ['/openmp']
    # -march=native is not applicable for cross-compilation
else:  # Unix-like systems
    openmp_compile_args = ['-fopenmp']
    openmp_link_args = ['-fopenmp']
    cpp_compile_args.extend(['-march=native', '-ffast-math', '-funroll-loops'])


# --- Define the extension ---

extensions = [
    Extension(
        # The name of the extension module, including the package path
        "ms_utils.formula_annotation.mass_decomposition_impl.mass_decomposer_cpp",
        [
            # List of source files with paths relative to the project root
            os.path.join(ext_base_path, "mass_decomposer_cpp.pyx"),
            os.path.join(ext_base_path, "mass_decomposer_common.cpp"),
            os.path.join(ext_base_path, "mass_decomposer_money_changing.cpp"),
            os.path.join(ext_base_path, "mass_decomposer_parallel.cpp")
        ],
        include_dirs=[numpy.get_include(), ext_base_path],
        extra_compile_args=cpp_compile_args + openmp_compile_args,
        extra_link_args=openmp_link_args,
        language="c++"
    )
]

# --- Main setup configuration ---

setup(
    # find_packages will discover your 'ms_utils' package in 'src'
    packages=find_packages(where="src"),
    # Tells setuptools that packages are in 'src'
    package_dir={"": "src"},
    # Cythonize the extensions
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
)