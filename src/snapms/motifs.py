import argparse
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
    DBMatch,
    H,
    assign_mass_cluster,
    build_molecular_families,
    find_db_matches,
    group_nodes,
    import_atlas,
    merge_duplicates,
)
from src.utils.cli import prepare_args
from src.utils.configs import SNAPMSConfig
from src.utils.cx import read_cx, write_cx
from src.utils.folders import prepare_directory
from src.utils.paths import ANNOTATION_STYLE_FILE
from src.utils.progress_bar import track
from src.utils.substructure import greedy_substructure_finder

logging.getLogger("matchms").setLevel(logging.ERROR)

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_SNAPMS_CONFIG = REPO / "config" / "snapms.yaml"


@dataclass
class Fragment:
    smiles: str
    origin: set[int]


def main():
    parser = argparse.ArgumentParser(description="motif fragment based SNAP-MS tool.")
    parser.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--graph",
        "-g",
        type=str,
        default=None,
        help="Target molecular network CX graph.",
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        default=None,
        help="Destination directory for output.",
    )
    parser.add_argument(
        "--reference-db",
        type=str,
        default=None,
        help="Reference structure database path.",
    )
    parser.add_argument(
        "--ppm-error", type=float, default=None, help="Mass tolerance error in PPM."
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Minimum network cluster size.",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=None,
        help="Maximum network cluster size.",
    )
    parser.add_argument(
        "--min-annotation-size",
        type=int,
        default=None,
        help="Minimum annotation cluster size.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Tanimoto similarity cutoff threshold.",
    )
    parser.add_argument(
        "--adduct-list", nargs="+", default=None, help="List of candidate adduct forms."
    )
    parser.add_argument(
        "--detect-adduct",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatically detect adduct annotations directly from graph properties.",
    )
    config = parser.parse_args()
    config = prepare_args(config, DEFAULT_SNAPMS_CONFIG)
    config = SNAPMSConfig(**dict(vars(config)))

    config.min_annotation_size = 2
    config.adduct_list = []
        
    config.display()
    fragment_snapms(config)


def fragment_snapms(config: SNAPMSConfig) -> None:
    atlas_con = import_atlas(config)

    file = Path(config.graph)

    mn = read_cx(str(file))

    annotation_folder = Path(config.result_folder)
    if annotation_folder.is_dir():
        shutil.rmtree(annotation_folder)
    if not Path(annotation_folder).exists():
        prepare_directory(annotation_folder)

    groups = group_nodes(mn)

    for identifier, nodes in track(groups.items(), description="running snap-ms..."):
        if len(nodes) < config.min_cluster_size:
            continue
        if len(nodes) > config.max_cluster_size:
            continue

        matches = find_fragment_matches(mn, nodes, config, atlas_con)
        G = build_molecular_families(matches, config)
        if G is None:
            continue

        values = {node: identifier for node in G.nodes}
        nx.set_node_attributes(G=G, values=values, name="mn_cluster_id")

        save_path = str(annotation_folder / f"graph-{identifier}.cx")
        write_cx(G, save_path, ANNOTATION_STYLE_FILE)


def find_fragment_matches(
    mn: nx.Graph, nodes: set, config: SNAPMSConfig, atlas_con: duckdb.DuckDBPyConnection
) -> list[DBMatch]:

    fragments = []
    for node in nodes:
        matches = find_db_matches(mn, {node}, config, atlas_con)
        fragments = merge_duplicates(fragments)
        motif = derive_motif(mn, node)

        for match in matches:
            for smiles in derive_valid_fragments(motif, [match]):
                fragment = DBMatch(**asdict(match))
                fragment.smiles = smiles
                fragments.append(fragment)

    fragments = assign_mass_cluster(fragments)

    return fragments


def derive_valid_fragments(motif: Spectrum, matches: list[DBMatch]) -> Generator:
    intensities = normalize_intensities(motif.peaks.intensities)
    valid_masses = np.abs(motif.peaks.mz[intensities > 0.3])

    for db_match in matches:
        adduct = ADDUCT_RULES[db_match.adduct]
        ion_mass = adduct.mass_shift

        mol = Chem.MolFromSmiles(db_match.smiles)
        if mol is None:
            continue

        for substructures in greedy_substructure_finder([mol]):
            for substructure in substructures:
                monoisotopic_mass = rdMolDescriptors.CalcExactMolWt(substructure)

                variations = [
                    monoisotopic_mass,
                    monoisotopic_mass + ion_mass,
                    monoisotopic_mass - H,
                    monoisotopic_mass - 2 * H,
                ]

                def valid(mass: float):
                    return np.any(np.isclose(valid_masses, mass, atol=1e-1))

                if not any(valid(v) for v in variations):
                    continue

                smiles = Chem.MolToSmiles(substructure)
                yield smiles


def derive_motif(G: nx.Graph, node: str | int) -> Spectrum:
    node_data = G.nodes[node]

    peaks_raw = node_data.get("motif_peaks_json", "[]")
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

        intensities = normalize_intensities(intensities)

    spectrum = Spectrum(
        mz=mz,
        intensities=intensities,
        metadata=metadata,
    )

    return spectrum


def normalize_intensities(intensities: np.ndarray) -> np.ndarray:
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


if __name__ == "__main__":
    main()
