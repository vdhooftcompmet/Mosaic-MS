import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, RDKFingerprint, AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.DataStructs import BulkDiceSimilarity
from rdkit.Chem import rdFingerprintGenerator


def similarity_matrix(smiles1: list[str], smiles2: list[str], fingerprint: str, matrix: str):
    fp1 = smiles_to_fingerprints(smiles1, fingerprint)
    fp2 = smiles_to_fingerprints(smiles2, fingerprint)
    mtrx = get_matrix(fp1, fp2, matrix)
    return mtrx


def cache(fn):
    fn_cache = {}
    def inner(smile, *args, **kwargs):
        if smile not in fn_cache:
            fn_cache[smile] = fn(smile, *args, **kwargs)
        return fn_cache[smile]
    return inner


MORGAN_GENERATORS = {}


def get_morgan_generator(radius=2, n_bits=2048):
    global MORGAN_GENERATORS
    key = radius, n_bits
    if key not in MORGAN_GENERATORS:
        MORGAN_GENERATORS[key] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return MORGAN_GENERATORS[key]

@cache
def smile_to_morgan_fp(smile, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smile)
    mfpgen = get_morgan_generator(radius=radius, n_bits=n_bits)
    fp = mfpgen.GetCountFingerprint(mol) 
    return fp


@cache
def smile_to_maccs_fp(smile: str):
    mol = Chem.MolFromSmiles(smile)
    return MACCSkeys.GenMACCSKeys(mol)


@cache
def smile_to_rdk_fp(smile: str):
    mol = Chem.MolFromSmiles(smile)
    return RDKFingerprint(mol)


def fp_tanimoto_similarity(fingerprints1, fingerprints2) -> np.ndarray:
    sim_matrix = np.zeros((len(fingerprints1), len(fingerprints2)))

    for i, fp in enumerate(fingerprints1):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fingerprints2)
        sim_matrix[i, :] = sims

    return sim_matrix


def fp_dice_similarity(fingerprints1, fingerprints2) -> np.ndarray:
    sim_matrix = np.zeros((len(fingerprints1), len(fingerprints2)))

    for i, fp in enumerate(fingerprints1):
        sims = BulkDiceSimilarity(fp, fingerprints2)
        sim_matrix[i, :] = sims

    return sim_matrix


def smiles_to_fingerprints(smiles: list[str], selected_fp_type: str) -> list[ExplicitBitVect]:
    match selected_fp_type.lower():
        case "maccs":       
            return [smile_to_maccs_fp(s) for s in smiles]
        case "morgan":
            return [smile_to_morgan_fp(s) for s in smiles]
        case "rdk":
            return [smile_to_rdk_fp(s) for s in smiles]
        case _:
            raise ValueError(f"unknown option {selected_fp_type}")
        

def get_matrix(fingerprints1, fingerprints2, selected_matrix_type: str):
    match selected_matrix_type.lower():
        case "dice":       
            return fp_dice_similarity(fingerprints1, fingerprints2)
        case "tanimoto":
            return fp_tanimoto_similarity(fingerprints1, fingerprints2)
        case _:
            raise ValueError(f"unknown option {selected_matrix_type}")
