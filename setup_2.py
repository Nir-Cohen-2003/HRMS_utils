import os
import sys
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
# Do not import numpy here at the top level

def get_extensions():
    """
    A deferred function to get the extension modules.
    This function is called only when setuptools is ready to build,
    ensuring that numpy and rdkit are available.
    """
    import numpy as np # Import numpy here
    
    # In a conda/pixi build environment, sys.prefix points to the root.
    env_prefix = sys.prefix
    rdkit_include_dir = os.path.join(env_prefix, 'include')
    rdkit_lib_dir = os.path.join(env_prefix, 'lib')

    # List of RDKit libraries to link against.
    rdkit_libs = [
        'RDKitGraphMol', 'RDKitSmilesParse', 'RDKitRDGeneral',
        'RDKitDataStructs', 'RDKitRDGeometryLib', 'boost_serialization'
    ]

    # Define the Cython extension
    # Paths are relative to the project root
    extensions = [
        Extension(
            "ms_utils.mces.fast_mol_filter.calculator",
            sources=[
                "src/ms_utils/mces/fast_mol_filter/fast_mol_filter/calculator.pyx",
                "src/ms_utils/mces/fast_mol_filter/fast_mol_filter/cpp_filter.cpp"
            ],
            include_dirs=[
                np.get_include(),
                rdkit_include_dir, # Add RDKit's include directory
                "src/ms_utils/mces/fast_mol_filter/fast_mol_filter"
            ],
            library_dirs=[rdkit_lib_dir], # Add RDKit's library directory
            libraries=rdkit_libs,
            language="c++",
            extra_compile_args=["-std=c++17", "-fopenmp", "-O3"],
            extra_link_args=["-fopenmp"],
        )
    ]
    return cythonize(extensions)

setup(
    name="ms_utils",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=get_extensions(),
)