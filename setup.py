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
def _is_clang():
    cc = (os.environ.get("CC","") + " " + os.environ.get("CXX","")).lower()
    if "clang" in cc:
        return True
    try:
        from distutils import ccompiler
        comp = ccompiler.new_compiler()
        comp.initialize()
        cmd0 = comp.compiler[0] if comp.compiler else ""
        if "clang" in cmd0.lower():
            return True
    except Exception:
        pass
    return False

is_windows = sys.platform.startswith('win')

if is_windows:
    if _is_clang():
        # clang (GNU-style driver) on Windows
        openmp_compile_args = ['-fopenmp']
        openmp_link_args = ['-fopenmp']
        cpp_compile_args = ['-std=c++17', '-O3']
    else:
        # MSVC
        openmp_compile_args = ['/openmp']
        openmp_link_args = []
else:
    openmp_compile_args = ['-fopenmp']
    openmp_link_args = ['-fopenmp']
    cpp_compile_args.extend([ '-ffast-math', '-funroll-loops'])


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