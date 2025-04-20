
from typing import List
from itertools import batched, chain
import joblib
from rdkit import Chem
from rdkit import RDLogger
import polars as pl

RDLogger.DisableLog('rdApp.*')

def sanitize_smiles_polars(smiles: List[str], batch_size: int = 10000) -> pl.Series:
    sanitized = sanitize_smiles(smiles, batch_size)
    return pl.Series(sanitized)


def sanitize_smiles(smiles: List[str], batch_size: int = 10000) -> List[str]:
    batches = list(batched(smiles, batch_size))
    sanitized = joblib.Parallel(n_jobs=-1)(joblib.delayed(_sanitize_smiles_batch)(batch) for batch in batches)
    return list(chain(*sanitized))


def _sanitize_smiles_batch(smiles: List[str]) -> List[str]:
    sanitized = []
    for s in smiles:
        try:
            mol = Chem.MolFromSmiles(s, sanitize=True)
            if mol is None:
                clean_smile = ''
            else:
                clean_smile = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        except Exception as e:
            print(f"Error sanitizing SMILES {s}: {e}")
            clean_smile = ''
        sanitized.append(clean_smile)
    return sanitized


def inchi_to_smiles_polars(inchi: List[str], batch_size: int = 10000) -> pl.Series:
    smiles = inchi_to_smiles_list(inchi, batch_size)
    return pl.Series(smiles)


def inchi_to_smiles_list(inchi_list: List[str], batch_size: int = 10000) -> List[str]:
    """Convert a list of InChI strings to SMILES."""
    batches = list(batched(inchi_list, batch_size))
    smiles_list = joblib.Parallel(n_jobs=-1)(joblib.delayed(_inchi_to_smiles_batch)(batch) for batch in batches)
    return list(chain(*smiles_list))


def _inchi_to_smiles_batch(inchi_list: List[str]) -> List[str]:
    """Convert a list of InChI strings to SMILES."""
    smiles_list = []
    for inchi in inchi_list:
        try:
            mol = Chem.MolFromInchi(inchi)
            if mol is None:
                smiles = ''
            else:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        except Exception as e:
            print(f"Error converting InChI {inchi}: {e}")
            smiles = ''
        smiles_list.append(smiles)
    return smiles_list


def smiles_to_inchi_polars(smiles: List[str], batch_size: int = 10000) -> pl.Series:
    inchi = smiles_to_inchi_list(smiles, batch_size)
    return pl.Series(inchi)


def smiles_to_inchi_list(smiles_list: List[str], batch_size: int = 10000) -> List[str]:
    """Convert a list of SMILES strings to InChI."""
    batches = list(batched(smiles_list, batch_size))
    inchi_list = joblib.Parallel(n_jobs=-1)(joblib.delayed(_smiles_to_inchi_batch)(batch) for batch in batches)
    return list(chain(*inchi_list))


def _smiles_to_inchi_batch(smiles_list: List[str]) -> List[str]:
    """Convert a list of SMILES strings to InChI."""
    inchi_list = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                inchi = ''
            else:
                inchi = Chem.MolToInchi(mol)
        except Exception as e:
            print(f"Error converting SMILES {smiles}: {e}")
            inchi = ''
        inchi_list.append(inchi)
    return inchi_list



