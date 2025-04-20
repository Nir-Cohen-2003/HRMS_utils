import dataclasses
from dataclasses import dataclass
from typing import List, Literal, Dict, Any, Optional, Union
from itertools import batched, chain
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
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
    fp_type: Literal['morgan', 'rdkit', 'atompair', 'torsion', 'maccs'] = 'morgan'
    # Determines the output type: bit vector (folded), sparse bit vector (unfolded),
    # count vector (folded), sparse count vector (unfolded).
    # Note: Output is always converted to a dense numpy array of size fpSize.
    # Note: fp_method is ignored for 'maccs'.
    fp_method: Literal['GetFingerprint', 'GetSparseFingerprint', 'GetCountFingerprint', 'GetSparseCountFingerprint'] = 'GetFingerprint'
    # Note: fpSize is ignored for 'maccs' (fixed size 167).
    fpSize: int = 2048
    # Morgan specific
    radius: Optional[int] = 4
    useBondTypes: Optional[bool] = True
    # RDKit specific
    minPath: Optional[int] = 1
    maxPath: Optional[int] = 7
    numBitsPerFeature: Optional[int] = 2
    # AtomPair specific
    use2D: Optional[bool] = True
    minDistance: Optional[int] = 1
    maxDistance: Optional[int] = 30
    countSimulation_AP: Optional[bool] = True  # Renamed, relevant for GetFingerprint
    includeChirality: Optional[bool] = False  # Common chirality flag (used by AtomPair, Torsion)
    # Torsion specific
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
        # Use inspect to get field names for robustness
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
    return pl.Series(flat_fps)


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

    # Ensure Python 3.12+ for batched, otherwise provide alternative or raise error
    try:
        from itertools import batched
    except ImportError:
        raise ImportError("itertools.batched requires Python 3.12+. Please update Python or use an alternative batching method.")

    batches = list(batched(smiles, batch_size))
    # Pass the validated FingerprintParams object to the batch function
    fps_batches = joblib.Parallel(n_jobs=-1)(joblib.delayed(_get_fp_batch)(batch, params_obj) for batch in batches)
    return fps_batches


def _get_fp_batch(smiles: List[str], fp_params: FingerprintParams) -> List[np.ndarray]:
    '''
    Generates fingerprints for a batch of SMILES strings based on FingerprintParams.

    Args:
        smiles (List[str]): List of SMILES strings for the batch.
        fp_params (FingerprintParams): Dataclass object with fingerprint parameters.

    Returns:
        List[np.ndarray]: List of generated fingerprints as dense numpy arrays.
    '''
    fpgen: Any = None
    method_kwargs = {}
    fp_method_func = None

    # Use match statement to configure generator and method kwargs
    match fp_params.fp_type:
        case 'morgan':
            morgan_args = {
                'radius': fp_params.radius,
                'fpSize': fp_params.fpSize,
                'useBondTypes': fp_params.useBondTypes,
                'atomInvariantsGenerator': fp_params.atomInvariantsGenerator,
                'includeChirality': fp_params.includeChirality  # Morgan supports chirality too
            }
            morgan_args = {k: v for k, v in morgan_args.items() if v is not None}
            fpgen = AllChem.GetMorganGenerator(**morgan_args)
        case 'rdkit':
            rdkit_args = {
                'minPath': fp_params.minPath,
                'maxPath': fp_params.maxPath,
                'fpSize': fp_params.fpSize,
                'numBitsPerFeature': fp_params.numBitsPerFeature
                # RDKit FP doesn't typically use includeChirality in generator
            }
            rdkit_args = {k: v for k, v in rdkit_args.items() if v is not None}
            fpgen = AllChem.GetRDKitFPGenerator(**rdkit_args)
        case 'atompair':
            atompair_args = {
                'minDistance': fp_params.minDistance,
                'maxDistance': fp_params.maxDistance,
                'includeChirality': fp_params.includeChirality,
                'use2D': fp_params.use2D,
            }
            atompair_args = {k: v for k, v in atompair_args.items() if v is not None}
            fpgen = AllChem.GetAtomPairGenerator(**atompair_args)
            if fp_params.fp_method == 'GetFingerprint':
                method_kwargs['fpSize'] = fp_params.fpSize
                method_kwargs['countSimulation'] = fp_params.countSimulation_AP
        case 'torsion':
            torsion_args = {
                'targetSize': fp_params.targetSize,
                'includeChirality': fp_params.includeChirality,
            }
            torsion_args = {k: v for k, v in torsion_args.items() if v is not None}
            fpgen = AllChem.GetTopologicalTorsionGenerator(**torsion_args)
            if fp_params.fp_method == 'GetFingerprint':
                method_kwargs['fpSize'] = fp_params.fpSize
                method_kwargs['countSimulation'] = fp_params.countSimulation_TT
        case 'maccs':
            # MACCS keys handled directly, no generator needed
            pass
        case _:  # Default case for unsupported types
            raise ValueError(f"Unsupported fingerprint type: {fp_params.fp_type}")

    # Get the fingerprint generation method function from the generator (if applicable)
    if fpgen is not None:
        try:
            fp_method_func = getattr(fpgen, fp_params.fp_method)
        except AttributeError:
            # Provide a more informative error if method is invalid for the generator
            valid_methods = [m for m in dir(fpgen) if m.startswith('Get') and 'Fingerprint' in m]
            raise ValueError(
                f"Unsupported fingerprint method '{fp_params.fp_method}' for generator type '{fp_params.fp_type}'. "
                f"Valid methods for {type(fpgen).__name__} might include: {valid_methods}"
            ) from None  # Suppress original AttributeError

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = []
    # Determine fingerprint length (MACCS has fixed size 167)
    fingerprint_length = 167 if fp_params.fp_type == 'maccs' else fp_params.fpSize

    for mol in mols:
        np_fp = np.zeros(fingerprint_length, dtype=np.float32)
        if mol is not None:
            try:
                fp = None
                # Generate the fingerprint
                match fp_params.fp_type:
                    case 'maccs':
                        fp = MACCSkeys.GenMACCSKeys(mol)
                    case _ if fp_method_func is not None:  # Use generator method for other types
                        fp = fp_method_func(mol, **method_kwargs)
                    case _:
                        # This case should ideally not be reached due to earlier checks
                        raise RuntimeError(f"Fingerprint generation function not determined for type {fp_params.fp_type}")

                # Convert RDKit fingerprint object to numpy array
                if fp is not None:
                    if isinstance(fp, DataStructs.ExplicitBitVect):
                        # Ensure np_fp has the correct size before conversion
                        if len(np_fp) != len(fp):
                            # This might happen if fpSize was set incorrectly for MACCS, adjust.
                            print(f"Warning: Adjusting numpy array size from {len(np_fp)} to {len(fp)} for {fp_params.fp_type}")
                            np_fp = np.zeros(len(fp), dtype=np.float32)
                        DataStructs.ConvertToNumpyArray(fp, np_fp)
                    elif hasattr(fp, 'GetNonzeroElements'):  # Handle sparse/count vectors
                        # Fold sparse/count vector into the fixed-size numpy array
                        for bit_id, count in fp.GetNonzeroElements().items():
                            if fingerprint_length > 0:
                                idx = bit_id % fingerprint_length
                                np_fp[idx] = count
                            else:
                                print(f"Warning: fingerprint_length is zero or invalid for folding. Skipping feature {bit_id}.")
                    else:
                        print(f"Warning: Unexpected fingerprint type {type(fp)} generated. Returning zeros.")
                # else: fp remains None, np_fp remains zeros if mol was valid but fp generation failed

            except Exception as e:
                # Consider logging the SMILES string that caused the error
                # smiles_str = Chem.MolToSmiles(mol) if mol else "Invalid Mol"
                print(f"Error generating fingerprint for a molecule: {e}. Returning zeros.")
                # np_fp remains zeros

        fps.append(np_fp)

    return fps