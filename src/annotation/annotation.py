import json
import logging
import shutil
from collections.abc import Generator
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import tomotopy as tp
from matchms import Spectrum
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.mn.mn import clean_mgf
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
from src.utils.motifs import run_overlap_scores_calculation
from src.utils.paths import MOTIF_FILE
from src.utils.progress_bar import track
from src.utils.substructure import greedy_substructure_finder

logging.getLogger("matchms").setLevel(logging.ERROR)


@dataclass
class Fragment:
    smiles: str
    origin: set[int]


def construct_motif_graph(params: AnnotationConfig) -> nx.Graph:
    model_path = Path(params.model).resolve()
    spectra = list(clean_mgf(str(params.mgf)))

    model = tp.LDAModel.load(str(model_path))
    result = run_overlap_scores_calculation(spectra, model)
    _beta_matrix, _phi_matrix, _theta_matrix, overlap_scores = result

    threshold = float(params.threshold)

    G = nx.Graph()

    for motif_index in range(overlap_scores.shape[0]):
        scores = overlap_scores[motif_index, :]

        assert len(scores) == len(spectra)

        above_threshold = np.where(scores > threshold)[0]
        top_n = np.argsort(scores)[-10:]
        spectrum_indices = np.intersect1d(above_threshold, top_n)

        motif = topic_to_motif(model, motif_index)
        motif_metadata = motif.to_dict()
        motif_metadata = {f"motif_{k}": str(v) for k, v in motif_metadata.items()}

        for spectrum_index in spectrum_indices:
            spectrum = spectra[spectrum_index]
            metadata = {k: str(v) for k, v in spectrum.to_dict().items()}

            # !! add cluster numbering based on motif index, not based on size
            metadata |= {"mn_cluster_id": motif_index}

            # !! add motif peaks for later use
            metadata |= motif_metadata

            G.add_node(f"m{motif_index}s{spectrum_index}", **metadata)

        for u, spectrum_index_u in enumerate(spectrum_indices):
            for v, spectrum_index_v in enumerate(spectrum_indices):
                if u <= v:
                    continue

                id_u = f"m{motif_index}s{spectrum_index_u}"
                id_v = f"m{motif_index}s{spectrum_index_v}"
                G.add_edge(id_u, id_v)

    return G


def topic_to_motif(model: tp.LDAModel, topic_index: int) -> Spectrum:
    mz, intensities = [], []

    topic_words = model.get_topic_words(topic_index, 50)

    for pair in topic_words:
        assert len(pair) == 2

        word, probability = pair
        mz_str = word[5:]
        mz.append(float(mz_str))

        if word.startswith("frag@"):
            intensities.append(probability)
        elif word.startswith("loss@"):
            intensities.append(probability * -1)
        else:
            raise ValueError()

    mz = np.array(mz)
    intensities = np.array(intensities)

    indices = np.argsort(mz)
    mz = mz[indices]
    intensities = intensities[indices]
    motif = Spectrum(np.array(mz), np.array(intensities), {"motif_id": topic_index})

    return motif


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

        matches = find_fragment_matches(
            Gsuper, nodes, config, atlas_con, break_rings=False
        )
        Gsub = build_molecular_families(matches, config)
        if Gsub is None:
            continue

        values = {node: identifier for node in Gsub.nodes}
        nx.set_node_attributes(G=Gsub, values=values, name="mn_cluster_id")

        save_path = str(annotation_folder / f"graph-{identifier}.cx")
        write_cx(Gsub, save_path, MOTIF_FILE)


def find_fragment_matches(
    mn: nx.Graph,
    nodes: set,
    config: AnnotationConfig,
    atlas_con: duckdb.DuckDBPyConnection,
    break_rings: bool,
) -> list[DBMatch]:

    fragments = []
    for node in nodes:
        matches = find_db_matches(mn, {node}, config, atlas_con)
        fragments = merge_duplicates(fragments)
        motif = derive_motif(mn, node)

        for match in matches:
            for mol, smiles, mz, variation, mass_error_da in derive_valid_fragments(
                motif, [match], break_rings
            ):
                morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=2048
                )

                fragment = DBMatch(**asdict(match))
                fragment.morgan_fingerprint = morgan_fp.ToBase64()
                fragment.inchikey = ""
                fragment.smiles = smiles
                fragment.mz = mz
                fragment.variation = variation
                fragment.mass_error_da = mass_error_da
                fragments.append(fragment)

    fragments = assign_mass_cluster(fragments)

    return fragments


def derive_valid_fragments(
    motif: Spectrum, matches: list[DBMatch], break_rings: bool
) -> Generator:
    intensities = motif.peaks.intensities
    intensities[intensities < 0] = 0
    n = np.max(intensities)
    if n != 0:
        intensities /= n

    valid_mz = motif.peaks.mz[intensities > 0.5]

    valid_masses = np.abs(valid_mz)

    for db_match in matches:
        adduct = ADDUCT_RULES[db_match.adduct]

        mol = Chem.MolFromSmiles(db_match.smiles)
        if mol is None:
            continue

        for substructure in greedy_substructure_finder([mol]):
            fragment_mass = rdMolDescriptors.CalcExactMolWt(substructure)

            for mz, variation, mass_error in filter_valid_mass(
                fragment_mass, adduct, valid_masses
            ):
                smiles = Chem.MolToSmiles(substructure)
                yield substructure, smiles, mz, variation, mass_error


def filter_valid_mass(fragment_mass, adduct: Adduct, valid_masses):
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
            yield mz, variation, mass_error
            return


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
