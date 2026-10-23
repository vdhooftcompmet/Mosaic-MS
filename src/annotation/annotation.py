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
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from src.ms2lda.ms2lda import spectra_to_documents
from src.snapms.snapms import (
    ADDUCT_RULES,
    Adduct,
    DBMatch,
    H,
    find_db_matches,
    group_nodes,
    import_atlas,
    merge_duplicates,
)
from src.utils.configs import AnnotationConfig
from src.utils.cx import read_cx, write_cx
from src.utils.folders import prepare_directory
from src.utils.paths import MOTIF_FILE
from src.utils.progress_bar import track

logging.getLogger("matchms").setLevel(logging.ERROR)


@dataclass
class MassMetadata:
    mz: float
    variation: str
    mass_error_da: float


@dataclass
class FragmentMetadata:
    mz: float
    variation: str
    mass_error_da: float
    smiles: str
    morgan_fingerprint: str
    mass_cluster: set[int]


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
        Gsub = _build_molecular_families(matches, config)
        if Gsub is None:
            continue

        values = {node: identifier for node in Gsub.nodes}
        nx.set_node_attributes(G=Gsub, values=values, name="mn_cluster_id")

        save_path = str(annotation_folder / f"graph-{identifier}.cx")
        write_cx(Gsub, save_path, MOTIF_FILE)


def build_spectra_graph(
    motifs: list[Spectrum], spectra: list[Spectrum], threshold: float
) -> nx.Graph:

    G = nx.Graph()

    for motif_index, spectrum_indices in enumerate(
        _detect_motifs(motifs, spectra, threshold)
    ):
        motif = motifs[motif_index]
        motif_metadata = motif.to_dict()
        motif_metadata = {f"motif_{k}": str(v) for k, v in motif_metadata.items()}
        # add nodes
        for spectrum_index in spectrum_indices:
            spectrum = spectra[spectrum_index].clone()

            metadata = {k: str(v) for k, v in spectrum.to_dict().items()}

            # !! add cluster numbering based on motif index, not based on size
            metadata |= {"mn_cluster_id": motif_index}

            # !! we need motif metadata later for annotation
            metadata |= motif_metadata

            G.add_node(f"m{motif_index}s{spectrum_index}", **metadata)

        # add edges
        for u, spectrum_index_u in enumerate(spectrum_indices):
            for v, spectrum_index_v in enumerate(spectrum_indices):
                if u <= v:
                    continue

                id_u = f"m{motif_index}s{spectrum_index_u}"
                id_v = f"m{motif_index}s{spectrum_index_v}"
                G.add_edge(id_u, id_v)
    return G


def _find_fragment_matches(
    mn: nx.Graph,
    nodes: set,
    config: AnnotationConfig,
    atlas_con: duckdb.DuckDBPyConnection,
) -> list[FragmentMetadata]:

    fragments = []
    for node in nodes:
        matches = find_db_matches(mn, {node}, config, atlas_con)
        matches = merge_duplicates(matches)
        motif = _derive_motif(mn, node)

        for match in matches:
            for fragment in _derive_valid_fragments(motif, match):
                fragments.append(fragment)

    fragments = _deduplicate_fragments(fragments)

    return fragments


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


def _derive_valid_fragments(motif: Spectrum, db_match: DBMatch) -> Generator:

    if len(motif.peaks.mz) == 0:
        return

    intensities = motif.peaks.intensities
    intensities[intensities < 0] = 0
    n = np.max(intensities)
    if n != 0:
        intensities /= n

    max_idx = np.argmax(intensities)
    valid_mass = motif.peaks.mz[max_idx]

    adduct = ADDUCT_RULES[db_match.adduct]

    mol = Chem.MolFromSmiles(db_match.smiles)
    if mol is None:
        return

    substructures = list(_greedy_substructure_finder(mol))
    for substructure in substructures:
        fragment_mass = rdMolDescriptors.CalcExactMolWt(substructure)

        for mass_metadata in _filter_valid_mass(fragment_mass, adduct, valid_mass):
            morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                substructure, radius=2, nBits=2048
            )
            fragment = FragmentMetadata(
                smiles=Chem.MolToSmiles(substructure),
                morgan_fingerprint=morgan_fp.ToBase64(),
                mass_cluster={int(db_match.neutral_mass)},
                **asdict(mass_metadata),
            )
            yield fragment


def _greedy_substructure_finder(mol: Chem.Mol, depth=2):
    yield mol
    if depth == 0:
        return

    n = mol.GetNumBonds()
    if n == 0:
        return

    is_in_ring = np.array([mol.GetBondWithIdx(i).IsInRing() for i in range(n)])
    is_aromatic = np.array([mol.GetBondWithIdx(i).GetIsAromatic() for i in range(n)])
    is_single = np.array(
        [mol.GetBondWithIdx(i).GetBondType() == Chem.BondType.SINGLE for i in range(n)]
    )

    valid_single_bonds = np.flatnonzero(~is_in_ring & ~is_aromatic & is_single)
    for index in valid_single_bonds:
        fragmented_mol = Chem.FragmentOnBonds(mol, [int(index)])
        fragments = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=False)

        for frag in fragments:
            res = _clean_and_check_fragment(frag)

            if res is None:
                continue

            yield res

            if depth >= 1:
                yield from _greedy_substructure_finder(frag, depth-1)


def _clean_and_check_fragment(frag: Chem.Mol):
    """Clears aromaticity, sanitizes, and deduplicates on-the-fly to save compute."""
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Atomic number 0 represents a dummy atom (*)
            atom.SetIsotope(
                0
            )  # This results in removal of the dummy atom number, helping us to merge identical SMILES later
    try:
        Chem.SanitizeMol(frag)
        return frag
    except Exception:
        return None


def _filter_valid_mass(fragment_mass: float, adduct: Adduct, valid_mass: float):

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
        mz = valid_mass
        mass_error_da = valid_mass - mass

        if np.abs(mass_error_da) <= 0.01:
            yield MassMetadata(mz, variation, mass_error_da)


def _deduplicate_fragments(fragments: list[FragmentMetadata]) -> list[FragmentMetadata]:
    seen: dict[str, FragmentMetadata] = {}

    for frag in fragments:
        existing = seen.get(frag.smiles)

        if existing is None:
            seen[frag.smiles] = frag

        else:
            for x in frag.mass_cluster:
                existing.mass_cluster.add(x)

    return list(seen.values())


def _build_molecular_families(
    matches: list[FragmentMetadata], config: AnnotationConfig
) -> nx.Graph | None:
    edges = _get_edges(matches, cutoff=config.cutoff)

    G = nx.Graph()
    G.add_nodes_from((i, asdict(match)) for i, match in enumerate(matches))
    G.add_edges_from(edges)

    _remove_small_subgraphs(G, config)  # small families are likely irrelevant
    _add_diversity(G, config)

    if len(G) == 0:  # empty graphs are not saved for ease of user investigation
        return None

    _add_cluster_numbering(G)

    return G


def _get_edges(
    matches: list[FragmentMetadata], cutoff: float = 0.66
) -> list[tuple[int, int]]:
    fingerprints = []
    for match in matches:
        fp = ExplicitBitVect(2048)
        fp.FromBase64(match.morgan_fingerprint)
        fingerprints.append(fp)

    dice_matrix = np.zeros(shape=(len(matches), len(matches)))
    for i, fp in enumerate(fingerprints):
        # we only calculate and fill in one triangle to the matrix to save compute
        similarity = DataStructs.BulkDiceSimilarity(fp, fingerprints[i + 1 :])
        dice_matrix[i, i + 1 :] = similarity

    rows, cols = np.where(np.triu(dice_matrix, k=1) > cutoff)

    result = []
    for u, v in zip(rows, cols):
        is_invalid = len(matches[u].mass_cluster | matches[v].mass_cluster) <= 1
        if (
            is_invalid
        ):  # edges between nodes representing the same single cluster are not drawn
            continue

        strength = dice_matrix[u, v]

        result.append((u, v, {"strength": strength}))

    return result


def _remove_small_subgraphs(graph: nx.Graph, config: AnnotationConfig) -> None:
    clusters = [x for x in nx.connected_components(graph)]
    for nodes in clusters:
        if len(nodes) > 1:
            continue

        node = next(iter(nodes))
        mass_cluster = graph.nodes[node]["mass_cluster"]
        if len(mass_cluster) > 1:
            continue

        graph.remove_nodes_from(nodes)


def _add_diversity(graph: nx.Graph, config: AnnotationConfig) -> None:
    clusters = [x for x in nx.connected_components(graph)]

    max_diversity = 0

    for nodes in clusters:
        diversity = set()

        for node in nodes:
            mass_cluster = graph.nodes[node]["mass_cluster"]
            diversity |= mass_cluster

        max_diversity = max(max_diversity, len(diversity))

        for node in nodes:
            graph.nodes[node]["diversity"] = len(diversity)

    for nodes in clusters:
        for node in nodes:
            is_top_candidate = graph.nodes[node]["diversity"] == max_diversity
            graph.nodes[node]["is_top_candidate"] = is_top_candidate


def _add_cluster_numbering(graph: nx.Graph) -> None:
    ordered_clusters = sorted(
        nx.connected_components(graph), key=lambda x: len(x), reverse=True
    )
    for cluster in ordered_clusters:
        i = _get_unique_id()
        for node in cluster:
            graph.nodes[node]["parent_cluster"] = i


ID_COUNTER = -1


def _get_unique_id() -> int:
    global ID_COUNTER
    ID_COUNTER += 1
    return ID_COUNTER


def _detect_motifs(motifs: list[Spectrum], spectra: list[Spectrum], threshold: float):
    overlap_scores = _calc_overlap_scores(motifs, spectra, True, 2)

    for motif_index in range(overlap_scores.shape[0]):
        scores = overlap_scores[motif_index, :]
        assert len(scores) == len(spectra)

        above_threshold = np.where(scores > threshold)[0]
        # top_n = np.argsort(scores)[-10:]
        # intersect = np.intersect1d(above_threshold, top_n)
        indices = [int(x) for x in above_threshold]
        yield indices


def _calc_overlap_scores(
    motifs: list[Spectrum],
    spectra: list[Spectrum],
    calc_losses: bool,
    significant_digits: int,
) -> np.ndarray:

    # 1. Convert both dataset spectra and consensus motifs into documents using your function
    spectra_docs = spectra_to_documents(spectra, calc_losses, significant_digits)
    motif_docs = spectra_to_documents(motifs, calc_losses, significant_digits)

    # 2. Build a unified vocabulary from all unique words found across both sets
    all_words = set()
    for doc in spectra_docs + motif_docs:
        if doc:
            all_words.update(doc)
    used_vocabs = sorted(list(all_words))
    word_to_idx = {word: idx for idx, word in enumerate(used_vocabs)}

    n_words = len(used_vocabs)
    n_docs = len(spectra)
    n_topics = len(motifs)

    # 3. Build beta_matrix (Topics x Words) from the consensus motif documents
    beta_matrix = np.zeros((n_topics, n_words), dtype=float)
    for i, doc in enumerate(motif_docs):
        if not doc:
            continue
        for word in doc:
            if word in word_to_idx:
                beta_matrix[i, word_to_idx[word]] += 1

    # Normalize each topic row to sum to 1 (proper probability distribution)
    row_sums = beta_matrix.sum(axis=1, keepdims=True)
    beta_matrix = np.divide(
        beta_matrix, row_sums, out=np.zeros_like(beta_matrix), where=row_sums != 0
    )

    # 4. Build word_counts matrix (Words x Docs) for the dataset spectra
    word_counts = np.zeros((n_words, n_docs), dtype=np.int32)
    for j, doc in enumerate(
        track(spectra_docs, description="processing spectra documents...")
    ):
        if not doc:
            continue
        for word in doc:
            if word in word_to_idx:
                word_counts[word_to_idx[word], j] += 1

    # 5. Calculate phi_matrix (Topics x Docs) -> topic distribution in every doc
    phi_matrix = beta_matrix @ word_counts
    phi_matrix /= phi_matrix.sum(axis=0, keepdims=True) + 1e-12

    # 6. Approximate theta_matrix (Topics x Docs) using vector projections (cosine similarity)
    doc_vectors = word_counts.astype(float)
    doc_norms = np.linalg.norm(doc_vectors, axis=0, keepdims=True) + 1e-12
    normalized_docs = doc_vectors / doc_norms

    motif_norms = np.linalg.norm(beta_matrix, axis=1, keepdims=True) + 1e-12
    normalized_motifs = beta_matrix / motif_norms

    theta_matrix = normalized_motifs @ normalized_docs
    theta_matrix = np.clip(theta_matrix, 0, None)  # Ensure non-negative
    theta_matrix /= theta_matrix.sum(axis=0, keepdims=True) + 1e-12

    # 7. Final overlap scores calculation
    overlap_scores = phi_matrix * theta_matrix

    return overlap_scores
