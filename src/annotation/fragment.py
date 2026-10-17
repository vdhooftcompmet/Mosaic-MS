import json
import logging
import shutil
from collections.abc import Generator
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
from matchms import Spectrum
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.snapms.snapms import (
    ADDUCT_RULES,
    Adduct,
    DBMatch,
    H,
    assign_mass_cluster,
    build_molecular_families,
    find_db_matches,
    group_nodes,
    import_atlas,
    merge_duplicates,
)
from src.utils.configs import AnnotationConfig
from src.utils.cx import read_cx, write_cx
from src.utils.folders import prepare_directory
from src.utils.paths import MOTIF_FILE
from src.utils.progress_bar import (
    track,  # Assuming you are using rich for progress bars
)

logging.getLogger("matchms").setLevel(logging.ERROR)


@dataclass
class Fragment:
    mol: Chem.Mol
    smiles: str
    mz: float
    variation: str
    mass_error_da: float


@dataclass
class Mass:
    mz: float
    variation: str
    error: float


def fragment_snapms(config: AnnotationConfig, graph=None) -> None:
    if graph is None:
        file = Path(config.graph)
        Gsuper = read_cx(str(file))
    else:
        Gsuper = graph

    atlas_con = import_atlas(config)

    annotation_folder = Path(config.result_folder)
    if annotation_folder.is_dir():
        shutil.rmtree(annotation_folder)
    if not Path(annotation_folder).exists():
        prepare_directory(annotation_folder)

    groups = group_nodes(Gsuper)

    for identifier, nodes in track(groups.items(), description="running snap-ms..."):
        if len(nodes) < config.min_cluster_size:
            continue
        if len(nodes) > config.max_cluster_size:
            continue

        matches = _find_fragment_matches(Gsuper, nodes, config, atlas_con)
        Gsub = build_molecular_families(matches, config)
        if Gsub is None:
            continue

        values = {node: identifier for node in Gsub.nodes}
        nx.set_node_attributes(G=Gsub, values=values, name="mn_cluster_id")

        save_path = str(annotation_folder / f"graph-{identifier}.cx")
        write_cx(Gsub, save_path, MOTIF_FILE)


def _find_fragment_matches(
    mn: nx.Graph,
    nodes: set,
    config: AnnotationConfig,
    atlas_con: duckdb.DuckDBPyConnection,
) -> list[DBMatch]:

    fragments = []
    for node in nodes:
        matches = find_db_matches(mn, {node}, config, atlas_con)

        print(f"{len(matches) = }")
        matches = merge_duplicates(matches)
        motif = _derive_motif(mn, node)

        for match in matches:
            for frag in _derive_valid_fragments(motif, match):
                morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    frag.mol, radius=2, nBits=2048
                )

                fragment = DBMatch(**asdict(match))
                fragment.morgan_fingerprint = morgan_fp.ToBase64()
                fragment.inchikey = ""
                fragment.smiles = frag.smiles
                fragment.mz = frag.mz
                fragment.variation = frag.variation
                fragment.mass_error_da = frag.mass_error_da
                fragments.append(fragment)

    print(f"{len(fragments) = }")
    fragments = assign_mass_cluster(fragments)

    return fragments


def _derive_valid_fragments(motif: Spectrum, db_match: DBMatch) -> Generator:
    print(f"Total motif peaks: {len(motif.peaks.mz)}")

    if len(motif.peaks.mz) == 0:
        return

    intensities = motif.peaks.intensities
    intensities[intensities < 0] = 0
    n = np.max(intensities)
    if n != 0:
        intensities /= n

    valid_mz = motif.peaks.mz[intensities > 0.5]
    print(f"Valid Mz after intensity filter (>0.5): {len(valid_mz)}")

    valid_masses = np.abs(valid_mz)

    adduct = ADDUCT_RULES[db_match.adduct]

    mol = Chem.MolFromSmiles(db_match.smiles)
    print(f"Base molecule loaded: {mol is not None}")
    if mol is None:
        return

    substructures = list(_greedy_substructure_finder([mol]))
    print(f"Substructures found: {len(substructures)}")
    for substructure in substructures:
        fragment_mass = rdMolDescriptors.CalcExactMolWt(substructure)

        for mass in _filter_valid_mass(fragment_mass, adduct, valid_masses):
            smiles = Chem.MolToSmiles(substructure)
            yield Fragment(substructure, smiles, mass.mz, mass.variation, mass.error)


# TODO
# probably the motif orderand cluster order aren't aligned, wrong filter applies, no results
def _filter_valid_mass(fragment_mass: float, adduct: Adduct, valid_masses: np.ndarray):
    DALTON_ERROR = 0.1

    variations = []
    for i in range(adduct.multiplier):
        n = i + 1
        variations += [
            # neutral fragment
            (f"[{n}M* + A]", fragment_mass + adduct.mass_shift),
            (f"[{n}M*]", fragment_mass),
            # fragment 'steals' H from other fragment
            (f"[{n}M+H + A]", fragment_mass + adduct.mass_shift + H),
            (f"[{n}M+H]", fragment_mass + H),
            # fragment 'donates' H to other fragment
            (f"[{n}M-H + A]", fragment_mass + adduct.mass_shift - H),
            (f"[{n}M-H]", fragment_mass - H),
        ]

    for variation, mass in variations:
        delta = valid_masses - mass
        i = np.argmin(np.abs(delta))

        mz = valid_masses[i]
        mass_error = delta[i]
        if np.abs(mass_error) <= DALTON_ERROR:
            yield Mass(mz, variation, mass_error)


def _derive_motif(G: nx.Graph, node: str | int) -> Spectrum:
    node_data = G.nodes[node]

    peaks_raw = node_data.get("motif_peaks_json")
    if peaks_raw is None:
        raise ValueError()

    peaks = json.loads(peaks_raw) if isinstance(peaks_raw, str) else peaks_raw

    metadata = {}
    for key, val in node_data.items():
        if not key.startswith("motif_"):
            continue

        clean_key = key[6:]
        if clean_key == "peaks_json":
            continue

        metadata[clean_key] = val

    if not peaks:
        mz = np.array([], dtype=float)
        intensities = np.array([], dtype=float)
    else:
        peaks_arr = np.array(peaks, dtype=float)
        mz = peaks_arr[:, 0]
        intensities = peaks_arr[:, 1]

        sort_indices = np.argsort(mz)
        mz = mz[sort_indices]
        intensities = intensities[sort_indices]

        intensities = _normalize_intensities(intensities)

    spectrum = Spectrum(
        mz=mz,
        intensities=intensities,
        metadata=metadata,
    )

    return spectrum


def _normalize_intensities(intensities: np.ndarray) -> np.ndarray:
    if len(intensities) == 0:
        return intensities

    max_intensity = intensities.max()
    if max_intensity == 0:
        return intensities
    return intensities / max_intensity


def clear_folder(folder_path: str | Path):
    folder_path = Path(folder_path)
    if folder_path.is_dir():
        for file in folder_path.iterdir():
            if file.is_file():
                file.write_text("")


def _greedy_substructure_finder(mols: list[Chem.Mol]):
    for mol in mols:
        yield mol

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
                res = _clean_and_check_fragment(frag)
                if res is not None:
                    yield res


def _clean_and_check_fragment(frag):
    """Clears aromaticity, sanitizes, and deduplicates on-the-fly to save compute."""
    try:
        Chem.SanitizeMol(frag)
        return frag
    except Exception:
        return None
