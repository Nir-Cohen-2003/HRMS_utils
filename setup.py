import os
import sys
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
# Do not import numpy here at the top level

# --- RDKit Configuration ---
def find_rdkit_paths():
    """
    Finds RDKit include and library paths, raising an error if they are not found.
    """
    # The specific header we need for compilation.
    required_header = os.path.join('rdkit', 'GraphMol', 'ROMol.h')

    # 1. Check for pixi/conda build environment variable
    if 'PREFIX' in os.environ:
        prefix_dir = os.environ['PREFIX']
        rdkit_inc_path = os.path.join(prefix_dir, 'include')
        rdkit_lib_path = os.path.join(prefix_dir, 'lib')
        if os.path.isdir(rdkit_inc_path) and os.path.isdir(rdkit_lib_path) and os.path.exists(os.path.join(rdkit_inc_path, required_header)):
            print(f"Found RDKit paths via PREFIX environment variable: {prefix_dir}")
            return rdkit_inc_path, rdkit_lib_path
            # 1b. Check for CONDA_PREFIX environment variable
    if 'CONDA_PREFIX' in os.environ:
        conda_prefix = os.environ['CONDA_PREFIX']
        rdkit_inc_path = os.path.join(conda_prefix, 'include')
        rdkit_lib_path = os.path.join(conda_prefix, 'lib')
        if os.path.isdir(rdkit_inc_path) and os.path.isdir(rdkit_lib_path) and os.path.exists(os.path.join(rdkit_inc_path, required_header)):
            print(f"Found RDKit paths via CONDA_PREFIX environment variable: {conda_prefix}")
            return rdkit_inc_path, rdkit_lib_path
    # 2. Check for user-set environment variable
    if 'RDKIT_DIR' in os.environ:
        rdkit_dir = os.environ['RDKIT_DIR']
        rdkit_inc_path = os.path.join(rdkit_dir, 'include')
        rdkit_lib_path = os.path.join(rdkit_dir, 'lib')
        if os.path.isdir(rdkit_inc_path) and os.path.isdir(rdkit_lib_path) and os.path.exists(os.path.join(rdkit_inc_path, required_header)):
            print(f"Found RDKit paths via RDKIT_DIR: {rdkit_dir}")
            return rdkit_inc_path, rdkit_lib_path
        else:
            print(f"Warning: RDKIT_DIR '{rdkit_dir}' is set but does not contain valid include/ and lib/ subdirectories with RDKit headers.", file=sys.stderr)

    # 3. If environment variables not set or invalid, try to find it automatically
    try:
        from rdkit import Chem
        import numpy as np # Import here, inside the try block
        rdkit_base = os.path.dirname(os.path.abspath(Chem.__file__))
        
        # Try to determine the environment prefix from the rdkit location
        env_prefix = os.path.abspath(os.path.join(rdkit_base, '..', '..', '..', '..'))

        # List of potential relative paths to the include/lib directories
        potential_paths = [
            (os.path.join(env_prefix, 'include'), os.path.join(env_prefix, 'lib')),
            (os.path.join(os.path.dirname(rdkit_base), 'include'), os.path.join(os.path.dirname(rdkit_base), 'lib')),
            (os.path.join(sys.prefix, 'include'), os.path.join(sys.prefix, 'lib')),
        ]

        for inc_path, lib_path in potential_paths:
            if os.path.isdir(inc_path) and os.path.isdir(lib_path) and os.path.exists(os.path.join(inc_path, required_header)):
                print(f"Found RDKit paths automatically: {inc_path}")
                return inc_path, lib_path

    except ImportError:
        pass # RDKit not importable, will fail below

    # 4. If all attempts fail, raise an informative error
    raise RuntimeError(
        "Could not find RDKit include directory.\n"
        "Please set the RDKIT_DIR environment variable to the root of your RDKit installation or run this build within a conda/pixi environment.\n"
        "For example: export RDKIT_DIR='/path/to/your/conda/env'\n"
        f"Make sure the path contains include/{required_header}"
    )

def get_extensions():
    """
    A deferred function to get the extension modules.
    This function is called only when setuptools is ready to build,
    ensuring that numpy and rdkit are available.
    """
    import numpy as np # Import numpy here
    
    try:
        rdkit_inc_path, rdkit_lib_path = find_rdkit_paths()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

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
                rdkit_inc_path,
                "src/ms_utils/mces/fast_mol_filter/fast_mol_filter"
            ],
            library_dirs=[rdkit_lib_path],
            runtime_library_dirs=[rdkit_lib_path],
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
    # Defer the extension building until setup is running
    ext_modules=get_extensions(),
)