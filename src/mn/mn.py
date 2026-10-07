from collections import namedtuple
from collections.abc import Callable
from pathlib import Path

import gensim
import networkx as nx
import numpy as np
from joblib import parallel_backend
from matchms import Spectrum, SpectrumProcessor
from matchms.filtering.default_pipelines import CLEAN_PEAKS, DEFAULT_FILTERS
from matchms.importing import load_from_mgf
from matchms.similarity.FlashSimilarity import FlashSimilarity
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from spec2vec import Spec2Vec

from src.utils.configs import MNConfig
from src.utils.context import suppress_output
from src.utils.progress_bar import track


def plain_similarity(
    spectra: list[Spectrum],
    config: MNConfig,
) -> np.ndarray:
    similarity_metric = get_similarity(
        config.similarity_type, config.flash_tolerance, config
    )
    with suppress_output():
        similarity_matrix = similarity_metric.matrix(
            list(spectra), list(spectra), array_type="numpy", is_symmetric=True
        )

    return similarity_matrix


def calculate_bootstrapping(
    spectra: list[Spectrum],
    config: MNConfig,
) -> tuple[np.ndarray, np.ndarray]:

    bins = global_bins(spectra, config.binning_decimals)
    binned_spectra = bin_spectra(spectra, config.binning_decimals)

    dataset_size = len(binned_spectra)
    similarity_metric = get_similarity(
        config.similarity_type, config.flash_tolerance, config
    )

    random_generator = np.random.default_rng(config.seed)

    total_pair_similarities = np.zeros((dataset_size, dataset_size), dtype=float)
    total_edge_support = np.zeros((dataset_size, dataset_size), dtype=float)

    print(config.B)
    for b in track(range(config.B), description="running specreboot..."):
        try:
            masked_spectra = _mask_spectra_globally(
                random_generator, bins, binned_spectra
            )

            with parallel_backend("loky", n_jobs=1):
                with suppress_output():
                    similarity_matrix = similarity_metric.matrix(
                        masked_spectra,
                        masked_spectra,
                        array_type="numpy",
                        is_symmetric=True,
                    )

            top_k_nearest_neighbours = mutual_topk(similarity_matrix, config.k)
            top_k_nearest_neighbours_binary = (top_k_nearest_neighbours != 0).astype(
                int
            )

            total_pair_similarities += similarity_matrix
            total_edge_support += top_k_nearest_neighbours_binary

        except Exception as e:
            print(f"\nCaught exception on iteration b={b}: {e}")
            raise e

    mean_similarities = total_pair_similarities / config.B

    np.fill_diagonal(
        mean_similarities, 1
    )  # needed to exaxtly match original implementation
    mean_edge_support = total_edge_support / config.B
    return mean_similarities, mean_edge_support


def global_bins(spectra: list[Spectrum], decimals: int) -> np.ndarray:
    rounded_spectra_values = [np.round(s.peaks.mz, decimals) for s in spectra]
    all_mz = np.concatenate(rounded_spectra_values)
    return np.sort(np.unique(all_mz))


def bin_spectra(spectra: list[Spectrum], decimals: int) -> list[Spectrum]:
    binned_spectra = []

    for spec in spectra:
        rounded_mz = np.round(spec.peaks.mz, decimals)
        intensities = spec.peaks.intensities.copy()
        metadata = spec.metadata.copy() if spec.metadata else None

        new_spectrum = Spectrum(rounded_mz, intensities, metadata)
        binned_spectra.append(new_spectrum)

    return binned_spectra


def mutual_topk(A: np.ndarray, k: int) -> np.ndarray:
    n = A.shape[0]
    A_work = A.copy()
    np.fill_diagonal(A_work, -np.inf)

    row_sorted = np.argsort(A_work, axis=1)[
        :, ::-1
    ]  # consider making this argpartition
    row_topk = row_sorted[:, :k]

    row_mask = np.zeros_like(A_work, dtype=bool)
    rows = np.arange(n)[:, None]
    row_mask[rows, row_topk] = True

    mutual_mask = row_mask & row_mask.T

    result = A.copy()
    result[~mutual_mask] = 0
    return result


def get_similarity(
    method_name: str, flash_tolerance: float, config: MNConfig
) -> FlashSimilarity | MS2DeepScore | Spec2Vec:

    match method_name:
        case "cos" | "cosine":
            return FlashSimilarity(
                score_type="cosine", matching_mode="fragment", tolerance=flash_tolerance
            )

        case "modcos" | "modified_cosine":
            return FlashSimilarity(
                score_type="cosine", matching_mode="hybrid", tolerance=flash_tolerance
            )

        case "ms2ds" | "ms2dp" | "ms2deepscore":
            if not Path(config.ms2deepscore_model_path).exists():
                raise FileNotFoundError(
                    f"file {config.ms2deepscore_model_path} not found"
                )

            ms2dp_model = load_model(str(config.ms2deepscore_model_path))
            return MS2DeepScore(ms2dp_model, progress_bar=False)

        case "s2v" | "spec2vec":
            if not Path(config.spec2vec_model_path).exists():
                raise FileNotFoundError(f"file {config.spec2vec_model_path} not found")

            w2v = gensim.models.Word2Vec.load(str(config.spec2vec_model_path))
            return Spec2Vec(
                model=w2v,
                intensity_weighting_power=0.5,
                allowed_missing_percentage=20.0,
                progress_bar=False,
            )

        case _:
            raise ValueError(f"unknown option {method_name}")


def _mask_spectra_globally(
    random_generator, global_bins: np.ndarray, binned_spectra: list[Spectrum]
) -> list[Spectrum]:
    result = []

    sampled_indices = random_generator.integers(
        0, len(global_bins), size=len(global_bins)
    )

    sampled_bins = global_bins[sampled_indices]
    sampled_bins = np.unique(sampled_bins)

    for index, spectrum in enumerate(binned_spectra):
        mask = np.isin(spectrum.peaks.mz, sampled_bins)

        mz = spectrum.peaks.mz[mask]
        intensities = spectrum.peaks.intensities[mask]

        mz = mz.astype("float32")
        intensities = intensities.astype("float32")

        if len(mz) == 0:
            mz = np.array([global_bins[0]], dtype="float32")
            intensities = np.array([0.0], dtype="float32")

        masked_spectrum = Spectrum(mz, intensities, spectrum.metadata)
        result.append(masked_spectrum)

    return result


EdgeData = namedtuple("EdgeData", ["u", "v", "sim", "sup", "lbl"])


def run_networking(
    spectra: list[Spectrum],
    similarity: np.ndarray,
    support: np.ndarray,
    network_type: str,
    config: MNConfig,
) -> nx.Graph:
    strat = {
        "base": filter_base_strategy(config.similarity_threshold),
        "threshold": filter_threshold_strategy(
            config.similarity_threshold, config.support_threshold
        ),
        "rescued": filter_rescue_strategy(
            config.similarity_threshold,
            config.support_threshold,
            config.rescue_similarity_threshold,
            config.support_threshold,
        ),
    }[network_type]

    similarity = np.nan_to_num(similarity, nan=0.0)
    support = np.nan_to_num(support, nan=0.0)

    G = build_graph(spectra, similarity, support, strat, config.max_component_size)
    return G


def filter_base_strategy(sim_threshold: float = 0.7) -> Callable[[EdgeData], EdgeData]:
    def inner(edge_data: EdgeData) -> EdgeData:
        mask = edge_data.sim >= sim_threshold
        edge_data = EdgeData(*(arr[mask] for arr in edge_data))
        return edge_data

    return inner


def filter_threshold_strategy(
    sim_threshold: float = 0.7, support_threshold: float = 0.3
) -> Callable[[EdgeData], EdgeData]:
    def inner(edge_data: EdgeData) -> EdgeData:
        mask = (edge_data.sim >= sim_threshold) & (edge_data.sup >= support_threshold)
        edge_data = EdgeData(*(arr[mask] for arr in edge_data))
        return edge_data

    return inner


def filter_rescue_strategy(
    sim_core: float = 0.7,
    support_core: float = 0.3,
    sim_rescue_min: float = 0.2,
    support_rescue: float = 0.4,
) -> Callable[[EdgeData], EdgeData]:
    def inner(edge_data: EdgeData) -> EdgeData:
        core_mask = (edge_data.sim >= sim_core) & (edge_data.sup >= support_core)

        rescue_mask = (
            (edge_data.sim >= sim_rescue_min)
            & (edge_data.sim < sim_core)
            & (edge_data.sup >= support_rescue)
        )
        mask = core_mask | rescue_mask

        labels = np.where(core_mask[mask], "core", "rescued")

        return EdgeData(
            edge_data.u[mask],
            edge_data.v[mask],
            edge_data.sim[mask],
            edge_data.sup[mask],
            labels,
        )

    return inner


def build_graph(
    spectra: list[Spectrum],
    sim: np.ndarray,
    sup: np.ndarray,
    filter_strategy: Callable[[EdgeData], EdgeData],
    max_component_size: int | None = None,
) -> nx.Graph:

    G, edge_data = _extract_graphdata(spectra, sim, sup)
    edge_data = filter_strategy(edge_data)

    if max_component_size is not None:
        edge_data = _filter_components(
            edge_data, max_component_size, retire_groups=True
        )

    for u, v, sim, sup, lbl in zip(*edge_data):
        metadata = dict(weight=float(sim), bootstrap_support=float(sup))
        if lbl != "":
            metadata |= dict(edge_class=str(lbl))
        G.add_edge(u, v, **metadata)

    return G


def _extract_graphdata(
    spectra: list[Spectrum], sim: np.ndarray, sup: np.ndarray
) -> tuple[nx.Graph, EdgeData]:
    G = nx.Graph()

    for index, spectrum in enumerate(spectra):
        metadata = {k: str(v) for k, v in spectrum.to_dict().items()}
        G.add_node(index, **metadata)

    n = len(spectra)

    # Extract upper-triangle pairs to avoid counting each edge twice.
    u, v = np.triu_indices(n, k=1)
    sim = sim[u, v]
    sup = sup[u, v]

    lbl = np.full(len(u), "")
    return G, EdgeData(u, v, sim, sup, lbl)


def _filter_components(
    edge_data: EdgeData,
    max_component_size: int,
    retire_groups: bool,
) -> EdgeData:
    if len(edge_data.u) == 0:
        return edge_data

    retired_groups = set()

    nr_of_nodes = max(np.max(edge_data.u), np.max(edge_data.v)) + 1
    node_groups = np.arange(
        nr_of_nodes
    )  # each node starts in its own singleton cluster
    group_sizes = np.ones(nr_of_nodes)  # every cluster starts with size 1

    # Work on a copy so the caller's array is not modified.
    sim = edge_data.sim.copy()

    # Process edges from strongest to weakest. When sim is equal, discriminate based on u and then v to make deterministic
    u_nodes = np.minimum(edge_data.u, edge_data.v)
    v_nodes = np.maximum(edge_data.v, edge_data.u)
    indices = np.lexsort((v_nodes, u_nodes, -sim))

    mask = np.zeros_like(sim)

    for i in indices:
        strength = sim[i]
        u, v = u_nodes[i], v_nodes[i]

        if strength == 0:  # all remaining edges are zeroed out, nothing left to do
            break

        u_group = node_groups[u]  # look up current cluster of u
        v_group = node_groups[v]  # look up current cluster of v

        if retire_groups and any(g in retired_groups for g in [u_group, v_group]):
            # At least one cluster is frozen; retire both to prevent partial absorption.
            retired_groups.add(u_group)
            retired_groups.add(v_group)
            continue

        if u_group == v_group:
            # Edge is within an existing cluster — no size change, always accept.
            mask[i] = 1
            continue

        u_group_size = group_sizes[u_group]
        v_group_size = group_sizes[v_group]

        if u_group_size + v_group_size > max_component_size:
            # Merging would exceed the limit; retire both clusters.
            retired_groups.add(u_group)
            retired_groups.add(v_group)
            continue

        # Accept the merge and update cluster bookkeeping.
        mask[i] = 1

        # The lower-numbered group absorbs the higher-numbered one.
        dominant_group, purged_group = sorted((u_group, v_group))
        node_groups[node_groups == purged_group] = dominant_group
        group_sizes[dominant_group] = u_group_size + v_group_size
        group_sizes[purged_group] = 0

    mask = mask.astype(bool)
    edge_data = EdgeData(*(arr[mask] for arr in edge_data))
    return edge_data


def add_cluster_numbering(graph: nx.Graph) -> None:
    ordered_clusters = sorted(
        nx.connected_components(graph), key=lambda x: len(x), reverse=True
    )
    for i, cluster in enumerate(ordered_clusters):
        for node in cluster:
            graph.nodes[node]["mn_cluster_id"] = i


def clean_mgf(path: Path | str) -> list[Spectrum]:
    assert isinstance(path, (str, Path)), "path must be a Path object or a string"

    spectra = list(load_from_mgf(path))
    spectrum_processor = SpectrumProcessor(DEFAULT_FILTERS + CLEAN_PEAKS)
    result, _ = spectrum_processor.process_spectra(spectra, progress_bar=False)

    for i, spectrum in enumerate(result):
        spectrum.set("spectrum_id", i)

    return result
