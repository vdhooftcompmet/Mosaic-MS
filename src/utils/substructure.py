from typing import Iterator

# def greedy_substructure_finder(mols: list[Chem.Mol], break_rings: bool = False):
#    for mol in mols:
#        for fragments in extensive_fragmentation(mol):
#            if fragments is None:
#                continue
#            yield [fragments]
#
#
# def generate_fragment_mols(
#    mol: Chem.Mol, replace_dummies: bool = True
# ) -> Iterator[Chem.Mol]:
#    """
#    Yields individual fragment Chem.Mol objects from an input Chem.Mol object.
#
#    Parameters:
#        mol: Input RDKit Mol object.
#        replace_dummies: If True, replaces dummy/wildcard atoms ([*]) with Hydrogens
#                         to yield clean, standard molecules. If False, keeps dummy atoms.
#    """
#    if mol is None:
#        return
#
#    # 1. Identify non-ring single bonds between heavy atoms
#    bonds_to_cleave = []
#    for bond in mol.GetBonds():
#        if (
#            not bond.IsInRing()
#            and bond.GetBondType() == Chem.BondType.SINGLE
#            and bond.GetBeginAtom().GetAtomicNum() > 1
#            and bond.GetEndAtom().GetAtomicNum() > 1
#        ):
#            bonds_to_cleave.append(bond.GetIdx())
#
#    # If no valid bonds to cleave, yield a copy of the original molecule
#    if not bonds_to_cleave:
#        yield Chem.Mol(mol)
#        return
#
#    # 2. Fragment the molecule (generates a single combined Mol with wildcard dummy atoms)
#    fragmented_mol = Chem.FragmentOnBonds(mol, bonds_to_cleave)
#
#    # 3. Iterate over disconnected fragment Mol objects and yield them
#    for frag in Chem.GetMolFrags(fragmented_mol, asMols=True):
#        if replace_dummies:
#            # Replace wildcard dummy atoms (atomic num 0) with hydrogen
#            rw_frag = Chem.RWMol(frag)
#            dummy_indices = [
#                atom.GetIdx() for atom in rw_frag.GetAtoms() if atom.GetAtomicNum() == 0
#            ]
#
#            # Remove dummy atoms in reverse order to preserve indexing
#            for idx in sorted(dummy_indices, reverse=True):
#                rw_frag.RemoveAtom(idx)
#
#            clean_frag = rw_frag.GetMol()
#            Chem.SanitizeMol(clean_frag)
#            yield clean_frag
#        else:
#            yield frag
#
#
# def generate_substructure(mol: Chem.Mol, break_rings: bool = False):
#
#    num_bonds = mol.GetNumBonds()
#    simple_bonds, complex_bonds = [], []
#
#    for i in range(num_bonds):
#        with Chem.RWMol(mol) as rwmol:
#            bond = rwmol.GetBondWithIdx(i)
#
#            if bond.GetBeginAtom().GetAtomicNum() <= 1:
#                continue
#            if bond.GetEndAtom().GetAtomicNum() <= 1:
#                continue
#
#            is_in_ring = bond.IsInRing()
#            if is_in_ring:
#                complex_bonds.append(i)
#                continue
#
#            is_single_bond = bond.GetBondType() == Chem.BondType.SINGLE
#            if not is_single_bond:
#                continue
#
#            simple_bonds.append(i)
#
#    for i in simple_bonds:
#        with Chem.RWMol(mol) as rwmol:
#            selected_bond = rwmol.GetBondWithIdx(i)
#            u = selected_bond.GetBeginAtomIdx()
#            v = selected_bond.GetEndAtomIdx()
#            rwmol.RemoveBond(u, v)
#
#        yield Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False)
#
#    if not break_rings:
#        return
#
#    for i in complex_bonds:
#        for ii in complex_bonds:
#            if i <= ii:
#                continue
#            with Chem.RWMol(mol) as rwmol:
#                b1 = rwmol.GetBondWithIdx(i)
#                b2 = rwmol.GetBondWithIdx(ii)
#                rwmol.RemoveBond(b1.GetBeginAtomIdx(), b1.GetEndAtomIdx())
#                rwmol.RemoveBond(b2.GetBeginAtomIdx(), b2.GetEndAtomIdx())
#
#            yield Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False)
import numpy as np
from rdkit import Chem


def greedy_substructure_finder(mols: list[Chem.Mol]):
    for mol in mols:
        n = mol.GetNumBonds()
        if n == 0:
            continue

        is_in_ring = np.array([mol.GetBondWithIdx(i).IsInRing() for i in range(n)])
        is_aromatic = np.array(
            [mol.GetBondWithIdx(i).GetIsAromatic() for i in range(n)]
        )
        is_single = np.array(
            [
                mol.GetBondWithIdx(i).GetBondType() == Chem.BondType.SINGLE
                for i in range(n)
            ]
        )

        valid_single_bonds = np.flatnonzero(~is_in_ring & ~is_aromatic & is_single)
        for index in valid_single_bonds:
            fragmented_mol = Chem.FragmentOnBonds(mol, [int(index)])
            fragments = Chem.GetMolFrags(
                fragmented_mol, asMols=True, sanitizeFrags=False
            )
            for frag in fragments:
                res = clean_and_check_fragment(frag)
                if res is not None:
                    yield res

        #
        #combinations = np.tril(np.ones((n, n), dtype=bool), k=-1)
        #
        # Mask out combinations with unbroken bonds
        #combinations[is_aromatic, :] = False
        #combinations[:, is_aromatic] = False
        #
        # Mask out cross-combinations of ring vs non-ring bonds
        #combinations[np.outer(is_in_ring, ~is_in_ring)] = False
        #combinations[np.outer(~is_in_ring, is_in_ring)] = False

        # Process double bond cuts
        # for bond_indices in np.argwhere(combinations):
        #    fragmented_mol = Chem.FragmentOnBonds(mol, [int(i) for i in bond_indices])
        #    fragments = Chem.GetMolFrags(
        #        fragmented_mol, asMols=True, sanitizeFrags=False
        #    )


#
#    for frag in fragments:
#        res = clean_and_check_fragment(frag, seen_smiles)
#        if res is not None:
#            yield res


def clean_and_check_fragment(frag):
    """Clears aromaticity, sanitizes, and deduplicates on-the-fly to save compute."""
    try:
        Chem.SanitizeMol(frag)
        return frag
    except Exception:
        return None
