from rdkit import Chem


def greedy_substructure_finder(mols: list[Chem.Mol], second_degree: bool = False):
    for mol in mols:
        for fragments in generate_substructure(mol):
            if fragments is None:
                continue
            yield fragments


def generate_substructure(
    mol: Chem.Mol
):

    num_bonds = mol.GetNumBonds()
    simple_bonds, complex_bonds = [], []

    for i in range(num_bonds):
        with Chem.RWMol(mol) as rwmol:
            selected_bond = rwmol.GetBondWithIdx(i)

            is_in_ring = selected_bond.IsInRing()

            if is_in_ring:
                complex_bonds.append(i)
            else:
                simple_bonds.append(i)

    for i in simple_bonds:
        with Chem.RWMol(mol) as rwmol:
            selected_bond = rwmol.GetBondWithIdx(i)
            u = selected_bond.GetBeginAtomIdx()
            v = selected_bond.GetEndAtomIdx()
            rwmol.RemoveBond(u, v)

        yield Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False)

    for i in complex_bonds:
        for ii in complex_bonds:
            if i <= ii:
                continue
            with Chem.RWMol(mol) as rwmol:
                b1 = rwmol.GetBondWithIdx(i)
                b2 = rwmol.GetBondWithIdx(ii)
                rwmol.RemoveBond(b1.GetBeginAtomIdx(), b1.GetEndAtomIdx())
                rwmol.RemoveBond(b2.GetBeginAtomIdx(), b2.GetEndAtomIdx())

            yield Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False)
