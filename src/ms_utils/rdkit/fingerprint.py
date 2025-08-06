import dataclasses
from dataclasses import dataclass
from typing import List, Literal, Dict, Any, Optional, Union, Iterable
from itertools import chain
from concurrent.futures import ProcessPoolExecutor  # Added ProcessPoolExecutor import
import functools  # Added functools import
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem, MACCSkeys, MolFromSmiles
from rdkit import RDLogger
from numpy.typing import NDArray
import numpy as np
import polars as pl

# RDLogger.DisableLog('rdApp.*')


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
    radius: Optional[int] = 2
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
        if self.fp_type == 'maccs':
            # MACCS keys have a fixed size of 167 bits
            self.fpSize = 167

    @classmethod
    def from_dict(cls, env: Dict[str, Any]):
        """Creates FingerprintParams instance from a dictionary, ignoring extra keys."""
        # Use inspect to get field names for robustness
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in env.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    def __eq__(self, other):
        """
        Determines if two FingerprintParams instances are equal by comparing all attributes.
        
        Args:
            other: Another FingerprintParams instance to compare with
            
        Returns:
            bool: True if all attributes are equal, False otherwise
        """
        if not isinstance(other, FingerprintParams):
            return False
        
        return all(
            getattr(self, field.name) == getattr(other, field.name)
            for field in dataclasses.fields(self)
        )


def get_fp_polars(smiles: Iterable[str], fp_params: Union[FingerprintParams, Dict[str, Any]] = FingerprintParams(), batch_size: int = 10000) -> pl.Series:
    """
    Generates fingerprints for a list of SMILES and returns them as a Polars Series.

    Args:
        smiles: List of SMILES strings.
        fp_params: A FingerprintParams object or a dictionary defining the fingerprint type and parameters.
                   Defaults to Morgan fingerprints (radius=4, size=2048).
        batch_size: Size of batches for parallel processing.

    Returns:
        A Polars Series containing the generated fingerprints as numpy arrays.
    
    Examples:
        >>> import polars as pl
        >>> import numpy as np
        
        >>> # Create FingerprintParams object with Morgan fingerprints, radius=2
        >>> fp_params = FingerprintParams(fp_type='morgan', radius=2, fpSize=2048)
        >>> smiles_series = pl.Series(['CCO', 'CCN', 'CCC'],dtype=pl.String)
        >>> result = get_fp_polars(smiles_series, fp_params)
        >>> isinstance(result, pl.Series)
        True
        >>> len(result)
        3
        >>> result[0].shape
        (2048,)
        >>> # Usage with map_batches
        >>> df = pl.DataFrame({'smiles': ['CCO', 'CCN', 'CCC']})
        >>> df = df.with_columns(
        ...     pl.col('smiles').map_batches(
        ...         lambda batch: get_fp_polars(batch, {'fp_type': 'maccs'})
        ...     ).alias('fingerprints')
        ... )
        >>> df['fingerprints'][0].shape
        (167,)
    """
    fps = get_fp_list(smiles, fp_params, batch_size)
    # Ensure fps is a flat list of numpy arrays before creating Series

    return pl.Series(fps)


def get_fp_list(smiles: Iterable[str], fp_params: Union[FingerprintParams, Dict[str, Any]] = FingerprintParams(), batch_size: int = 10000) -> List[NDArray]:
    """
    Generates fingerprints for a list of SMILES in parallel batches.

    Args:
        smiles: List of SMILES strings.
        fp_params: A FingerprintParams object or a dictionary defining the fingerprint type and parameters.
                   Defaults to Morgan fingerprints (radius=4, size=2048).
        batch_size: Size of batches for parallel processing.

    Returns:
        A list containing the fingerprints (numpy arrays).
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
    # Pass the validated FingerprintParams object to the batch function using partial
    partial_get_fp_batch = functools.partial(_get_fp_batch, fp_params=params_obj)

    fps_batches = []
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # map returns an iterator, convert it to a list
        fps_batches = list(executor.map(partial_get_fp_batch, batches))

    flat_fps = list(chain.from_iterable(fps_batches))
    return flat_fps


def _get_fp_batch(smiles: List[str], fp_params: FingerprintParams) -> List[NDArray]:
    '''
    Generates fingerprints for a batch of SMILES strings based on FingerprintParams.

    Args:
        smiles (List[str]): List of SMILES strings for the batch.
        fp_params (FingerprintParams): Dataclass object with fingerprint parameters.

    Returns:
        List[NDArray]: List of generated fingerprints as dense numpy arrays.
    '''
    RDLogger.DisableLog('rdApp.*')  # type: ignore[call-arg]

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
            fpgen = Chem.GetMorganGenerator(**morgan_args)
        case 'rdkit':
            rdkit_args = {
                'minPath': fp_params.minPath,
                'maxPath': fp_params.maxPath,
                'fpSize': fp_params.fpSize,
                'numBitsPerFeature': fp_params.numBitsPerFeature
                # RDKit FP doesn't typically use includeChirality in generator
            }
            rdkit_args = {k: v for k, v in rdkit_args.items() if v is not None}
            fpgen = Chem.GetRDKitFPGenerator(**rdkit_args)
        case 'atompair':
            atompair_args = {
                'minDistance': fp_params.minDistance,
                'maxDistance': fp_params.maxDistance,
                'includeChirality': fp_params.includeChirality,
                'use2D': fp_params.use2D,
            }
            atompair_args = {k: v for k, v in atompair_args.items() if v is not None}
            fpgen = Chem.GetAtomPairGenerator(**atompair_args)
            if fp_params.fp_method == 'GetFingerprint':
                method_kwargs['fpSize'] = fp_params.fpSize
                method_kwargs['countSimulation'] = fp_params.countSimulation_AP
        case 'torsion':
            torsion_args = {
                'targetSize': fp_params.targetSize,
                'includeChirality': fp_params.includeChirality,
            }
            torsion_args = {k: v for k, v in torsion_args.items() if v is not None}
            fpgen = Chem.GetTopologicalTorsionGenerator(**torsion_args)
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

    mols = [MolFromSmiles(s) for s in smiles]
    fps = []

    for mol in mols:
        np_fp = np.zeros(fp_params.fpSize, dtype=np.float32)
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
                            if fp_params.fpSize > 0:
                                idx = bit_id % fp_params.fpSize
                                np_fp[idx] = count
                            else:
                                print(f"Warning: fingerprint_length is zero or invalid for folding. Skipping feature {bit_id}.")
                    else:
                        print(f"Warning: Unexpected fingerprint type {type(fp)} generated. Returning zeros.")
                # else: fp remains None, np_fp remains zeros if mol was valid but fp generation failed

            except Exception as e:
                # Consider logging the SMILES string that caused the error
                print(f"Error generating fingerprint for a molecule: {e}. Returning zeros.")
                # np_fp remains zeros

        fps.append(np_fp)

    return fps

if __name__ == "__main__":
    # test the doctest of get_fp_polars
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    # Example usage