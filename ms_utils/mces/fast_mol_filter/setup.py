from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys

# Get the absolute path of the directory containing this setup.py file
SETUP_DIR = os.path.abspath(os.path.dirname(__file__))

# --- RDKit Configuration ---
def find_rdkit_paths():
    """
    Finds RDKit include and library paths, raising an error if they are not found.
    """
    # The specific header we need for compilation.
    # This is more reliable than checking for a generic 'rdkit.h'.
    required_header = os.path.join('rdkit', 'GraphMol', 'ROMol.h')

    # 1. Check for environment variable
    if 'RDKIT_DIR' in os.environ:
        rdkit_dir = os.environ['RDKIT_DIR']
        rdkit_inc_path = os.path.join(rdkit_dir, 'include')
        rdkit_lib_path = os.path.join(rdkit_dir, 'lib')
        if os.path.isdir(rdkit_inc_path) and os.path.isdir(rdkit_lib_path) and os.path.exists(os.path.join(rdkit_inc_path, required_header)):
            print(f"Found RDKit paths via RDKIT_DIR: {rdkit_dir}")
            return rdkit_inc_path, rdkit_lib_path
        else:
            print(f"Warning: RDKIT_DIR '{rdkit_dir}' is set but does not contain valid include/ and lib/ subdirectories with RDKit headers.")

    # 2. If environment variable not set or invalid, try to find it automatically
    try:
        from rdkit import Chem
        rdkit_base = os.path.dirname(os.path.abspath(Chem.__file__))
        
        # Try to determine the environment prefix from the rdkit location
        # This is more robust than sys.prefix in layered environments (like pixi)
        env_prefix = os.path.abspath(os.path.join(rdkit_base, '..', '..', '..', '..'))

        # List of potential relative paths to the include/lib directories
        potential_paths = [
            # Common structure for conda/pixi environments
            (os.path.join(env_prefix, 'include'), os.path.join(env_prefix, 'lib')),
            # Standard structure for rdkit-pypi wheel
            (os.path.join(os.path.dirname(rdkit_base), 'include'), os.path.join(os.path.dirname(rdkit_base), 'lib')),
            # Fallback to sys.prefix (might be different from the actual rdkit env)
            (os.path.join(sys.prefix, 'include'), os.path.join(sys.prefix, 'lib')),
        ]

        for inc_path, lib_path in potential_paths:
            # Check for the existence of a key RDKit header
            if os.path.isdir(inc_path) and os.path.isdir(lib_path) and os.path.exists(os.path.join(inc_path, required_header)):
                print(f"Found RDKit paths automatically: {inc_path}")
                return inc_path, lib_path

    except ImportError:
        pass # RDKit not importable, will fail below

    # 3. If all attempts fail, raise an informative error
    raise RuntimeError(
        "Could not find RDKit include directory.\n"
        f"Please set the RDKIT_DIR environment variable to the root of your RDKit installation.\n"
        "For example: export RDKIT_DIR='/path/to/your/conda/env'\n"
        f"Make sure the path contains include/{required_header}\n"
        "You can find the path by running: python -c 'import rdkit; import os; print(os.path.abspath(os.path.join(os.path.dirname(rdkit.__file__), \"..\", \"..\", \"..\", \"..\")))'"
    )

try:
    rdkit_inc_path, rdkit_lib_path = find_rdkit_paths()
except (ImportError, RuntimeError) as e:
    print(f"Error: {e}")
    sys.exit(1)


# List of RDKit libraries to link against.
rdkit_libs = [
    'GraphMol',
    'SmilesParse',
    'RDGeneral',
    'DataStructs',
    'RDGeometryLib',
    'boost_serialization'
]

extensions = [
    Extension(
        "fast_mol_filter.calculator",
        sources=[
            os.path.join(SETUP_DIR, "fast_mol_filter/calculator.pyx"),
            os.path.join(SETUP_DIR, "fast_mol_filter/cpp_filter.cpp")
        ],
        include_dirs=[
            np.get_include(), # Add numpy headers
            rdkit_inc_path,   # Add main include dir for RDKit and Boost
            os.path.join(rdkit_inc_path, 'rdkit'), # e.g., /path/to/env/include/rdkit
            os.path.join(SETUP_DIR, "fast_mol_filter")
        ],
        library_dirs=[
            rdkit_lib_path    # Tell the linker where to find libraries
        ],
        runtime_library_dirs=[
            rdkit_lib_path    # Embed the library path for runtime linking
        ],
        libraries=rdkit_libs, # Tell the linker which libraries to link
        language="c++",
        extra_compile_args=["-std=c++17", "-fopenmp", "-O3"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="fast_mol_filter",
    version="0.1.0",
    packages=["fast_mol_filter"],
    ext_modules=cythonize(extensions),
    install_requires=[
        "numpy",
        "cython"
    ],
    zip_safe=False,
)