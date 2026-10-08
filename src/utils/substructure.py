from rdkit import Chem


def greedy_substructure_finder(mols: list[Chem.Mol]):
    for mol in mols:
        num_bonds = mol.GetNumBonds()
        bonds_index = list(range(num_bonds))

        for bond_index in bonds_index:
            fragments = generate_substructure(mol, bond_index)

            if fragments is None:
                continue

            smiles1, smiles2 = fragments
            yield smiles1, smiles2


def generate_substructure(
    mol: Chem.Mol, bond_index: int
) -> None | tuple[Chem.Mol, Chem.Mol]:

    with Chem.RWMol(mol) as rwmol:
        selected_bond = rwmol.GetBondWithIdx(bond_index)

        is_not_single_bond = selected_bond.GetBondType() != Chem.BondType.SINGLE
        is_in_ring = selected_bond.IsInRing()

        if is_not_single_bond or is_in_ring:
            return None

        u = selected_bond.GetBeginAtomIdx()
        v = selected_bond.GetEndAtomIdx()
        rwmol.RemoveBond(u, v)

    frags = Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False)
    if len(frags) != 2:
        return None

    frag1, frag2 = frags
    return frag1, frag2
