import dataclasses
from dataclasses import dataclass
from typing import List, Literal, Dict, Any, Optional, Union
from itertools import batched, chain
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys  # Added MACCSkeys for potential future use
from rdkit import RDLogger
import numpy as np
import polars as pl

RDLogger.DisableLog('rdApp.*')


@dataclass
class FingerprintParams:
    """
    Dataclass to hold parameters for fingerprint generation.

    Allows specifying the fingerprint type and its associated parameters.
    Can be initialized with a dictionary.
    """
    fp_type: Literal['morgan', 'rdkit', 'atompair', 'torsion'] = 'morgan'
    # Determines the output type: bit vector (folded), sparse bit vector (unfolded),
    # count vector (folded), sparse count vector (unfolded).
    # Note: Output is always converted to a dense numpy array of size fpSize.
    fp_method: Literal['GetFingerprint', 'GetSparseFingerprint', 'GetCountFingerprint', 'GetSparseCountFingerprint'] = 'GetFingerprint'
    fpSize: int = 2048
    # Morgan specific
    radius: Optional[int] = 4
    useBondTypes: Optional[bool] = True
    # RDKit specific
    minPath: Optional[int] = 1
    maxPath: Optional[int] = 7
    numBitsPerFeature: Optional[int] = 2
    # AtomPair specific
    includeChirality_AP: Optional[bool] = False  # Renamed to avoid clash
    use2D: Optional[bool] = True
    minDistance: Optional[int] = 1
    maxDistance: Optional[int] = 30
    countSimulation_AP: Optional[bool] = True  # Renamed, relevant for GetFingerprint
    # Torsion specific
    includeChirality_TT: Optional[bool] = False  # Renamed to avoid clash
    targetSize: Optional[int] = 4
    countSimulation_TT: Optional[bool] = True  # Renamed, relevant for GetFingerprint
    # Common / Advanced
    atomInvariantsGenerator: Optional[Any] = None  # For Morgan features, etc. Requires RDKit objects

    def __post_init__(self):
        # Ensure radius is set for morgan if not provided
        if self.fp_type == 'morgan' and self.radius is None:
            self.radius = 4  # Default Morgan radius

    @classmethod
    def from_dict(cls, env: Dict[str, Any]):
        """Creates FingerprintParams instance from a dictionary, ignoring extra keys."""
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in env.items() if k in valid_keys}
        return cls(**filtered_dict)


def get_fp_polars(smiles: List[str], fp_params: Union[FingerprintParams, Dict[str, Any]] = FingerprintParams(), batch_size: int = 10000) -> pl.Series:
    """
    Generates fingerprints for a list of SMILES and returns them as a Polars Series.

    Args:
        smiles: List of SMILES strings.
        fp_params: A FingerprintParams object or a dictionary defining the fingerprint type and parameters.
                   Defaults to Morgan fingerprints (radius=4, size=2048).
        batch_size: Size of batches for parallel processing.

    Returns:
        A Polars Series containing the generated fingerprints as numpy arrays.
    """
    fps = get_fp_list(smiles, fp_params, batch_size)
    # Ensure fps is a flat list of numpy arrays before creating Series
    flat_fps = list(chain.from_iterable(fps))
    return pl.Series(flat_fps)  # Changed from chain(*fps) which might not work correctly if fps is list of lists


def get_fp_list(smiles: List[str], fp_params: Union[FingerprintParams, Dict[str, Any]] = FingerprintParams(), batch_size: int = 10000) -> List[List[np.ndarray]]:
    """
    Generates fingerprints for a list of SMILES in parallel batches.

    Args:
        smiles: List of SMILES strings.
        fp_params: A FingerprintParams object or a dictionary defining the fingerprint type and parameters.
                   Defaults to Morgan fingerprints (radius=4, size=2048).
        batch_size: Size of batches for parallel processing.

    Returns:
        A list of lists, where each inner list contains the fingerprints (numpy arrays) for a batch.
    """
    if isinstance(fp_params, dict):
        params_obj = FingerprintParams.from_dict(fp_params)
    else:
        params_obj = fp_params

    batches = list(batched(smiles, batch_size))
    # Pass the validated FingerprintParams object to the batch function
    fps_batches = joblib.Parallel(n_jobs=-1)(joblib.delayed(_get_fp_batch)(batch, params_obj) for batch in batches)
    return fps_batches  # Returns list of lists


def _get_fp_batch(smiles: List[str], fp_params: FingerprintParams) -> List[np.ndarray]:
    '''
    Generates fingerprints for a batch of SMILES strings based on FingerprintParams.

    Args:
        smiles (List[str]): List of SMILES strings for the batch.
        fp_params (FingerprintParams): Dataclass object with fingerprint parameters.

    Returns:
        List[np.ndarray]: List of generated fingerprints as dense numpy arrays.
    '''
    fpgen: Any = None  # Fingerprint generator object
    method_kwargs = {}  # Arguments specific to the fp_method call

    # Create the appropriate generator based on fp_type
    if fp_params.fp_type == 'morgan':
        morgan_args = {
            'radius': fp_params.radius,  # Already defaulted in __post_init__
            'fpSize': fp_params.fpSize,
            'useBondTypes': fp_params.useBondTypes,
            'atomInvariantsGenerator': fp_params.atomInvariantsGenerator
        }
        morgan_args = {k: v for k, v in morgan_args.items() if v is not None}
        fpgen = AllChem.GetMorganGenerator(**morgan_args)
    elif fp_params.fp_type == 'rdkit':
        rdkit_args = {
            'minPath': fp_params.minPath,
            'maxPath': fp_params.maxPath,
            'fpSize': fp_params.fpSize,
            'numBitsPerFeature': fp_params.numBitsPerFeature
        }
        rdkit_args = {k: v for k, v in rdkit_args.items() if v is not None}
        fpgen = AllChem.GetRDKitFPGenerator(**rdkit_args)
    elif fp_params.fp_type == 'atompair':
        atompair_args = {
            'minDistance': fp_params.minDistance,
            'maxDistance': fp_params.maxDistance,
            'includeChirality': fp_params.includeChirality_AP,
            'use2D': fp_params.use2D,
        }
        atompair_args = {k: v for k, v in atompair_args.items() if v is not None}
        fpgen = AllChem.GetAtomPairGenerator(**atompair_args)
        # fpSize and countSimulation are passed to GetFingerprint method for AtomPair
        if fp_params.fp_method == 'GetFingerprint':
            method_kwargs['fpSize'] = fp_params.fpSize
            method_kwargs['countSimulation'] = fp_params.countSimulation_AP
    elif fp_params.fp_type == 'torsion':
        torsion_args = {
            'targetSize': fp_params.targetSize,
            'includeChirality': fp_params.includeChirality_TT,
        }
        torsion_args = {k: v for k, v in torsion_args.items() if v is not None}
        fpgen = AllChem.GetTopologicalTorsionGenerator(**torsion_args)
        # fpSize and countSimulation are passed to GetFingerprint method for Torsion
        if fp_params.fp_method == 'GetFingerprint':
            method_kwargs['fpSize'] = fp_params.fpSize
            method_kwargs['countSimulation'] = fp_params.countSimulation_TT
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_params.fp_type}")

    # Get the fingerprint generation method function from the generator
    try:
        fp_method_func = getattr(fpgen, fp_params.fp_method)
    except AttributeError:
        raise ValueError(f"Unsupported fingerprint method '{fp_params.fp_method}' for generator type '{fp_params.fp_type}'")

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = []
    fingerprint_length = fp_params.fpSize  # Use fpSize for numpy array dimension

    for mol in mols:
        np_fp = np.zeros(fingerprint_length, dtype=np.float32)  # Initialize output array
        if mol is not None:
            try:
                # Generate the fingerprint using the selected method and specific kwargs
                fp = fp_method_func(mol, **method_kwargs)

                # Convert RDKit fingerprint object to numpy array
                if isinstance(fp, DataStructs.ExplicitBitVect):
                    DataStructs.ConvertToNumpyArray(fp, np_fp)
                elif hasattr(fp, 'GetNonzeroElements'):  # Handle sparse/count vectors
                    # Fold sparse/count vector into the fixed-size numpy array
                    for bit_id, count in fp.GetNonzeroElements().items():
                        idx = bit_id % fingerprint_length
                        np_fp[idx] = count  # Use count for count vectors, implicitly 1 for sparse bit vectors
                else:
                    # Fallback/Warning for unexpected types
                    print(f"Warning: Unexpected fingerprint type {type(fp)} generated. Returning zeros.")

            except Exception as e:
                # Catch potential errors during fingerprint generation for a specific molecule
                print(f"Error generating fingerprint for a molecule: {e}. Returning zeros.")
                # np_fp remains zeros

        # Append the (potentially zero) numpy array for this molecule
        fps.append(np_fp)

    return fps