import numpy as np
import networkx as nx
from collections import namedtuple
from typing import Callable
from utils.configs import MNConfig
from matchms import Spectrum

EdgeData = namedtuple("EdgeData", ["u", "v", "sim", "sup", "lbl"])


def run_networking(
    spectra: list[Spectrum],
    similarity: np.ndarray,
    support: np.ndarray,
    network_type: str,
    config: MNConfig
) -> nx.Graph:
    strat = {
        "base"      : filter_base_strategy(
            config.similarity_threshold
        ),
        "threshold" : filter_threshold_strategy(
            config.similarity_threshold,
            config.support_threshold
        ),
        "rescued"   : filter_rescue_strategy(
            config.similarity_threshold,
            config.support_threshold,
            config.rescue_similarity_threshold,
            config.support_threshold
        ),
    }[network_type]

    similarity = np.nan_to_num(similarity, nan=0.0)
    support = np.nan_to_num(support, nan=0.0)

    G = build_graph(spectra, similarity, support, strat, config.max_component_size)
    return G


def filter_base_strategy(sim_threshold: float = 0.7) -> Callable[[EdgeData], EdgeData]:
    def inner(edge_data: EdgeData) -> EdgeData:
        mask = edge_data.sim  >= sim_threshold
        edge_data = EdgeData(*(arr[mask] for arr in edge_data))
        return edge_data
    return inner


def filter_threshold_strategy(sim_threshold: float = 0.7, support_threshold: float = 0.3) -> Callable[[EdgeData], EdgeData]:
    def inner(edge_data: EdgeData) -> EdgeData:
        mask = (edge_data.sim >= sim_threshold) & (edge_data.sup >= support_threshold)
        edge_data = EdgeData(*(arr[mask] for arr in edge_data))
        return edge_data
    return inner


def filter_rescue_strategy(
    sim_core: float = 0.7,
    support_core: float = 0.3,
    sim_rescue_min: float = 0.2,
    support_rescue: float = 0.4
) -> Callable[[EdgeData], EdgeData]:
    def inner(edge_data: EdgeData) -> EdgeData:
        core_mask   = (edge_data.sim >= sim_core) & (edge_data.sup >= support_core)

        rescue_mask = (edge_data.sim >= sim_rescue_min) & (edge_data.sim < sim_core) & (edge_data.sup >= support_rescue)
        mask = core_mask | rescue_mask

        labels = np.where(core_mask[mask], "core", "rescued")

        return EdgeData(
            edge_data.u[mask],
            edge_data.v[mask],
            edge_data.sim[mask],
            edge_data.sup[mask],
            labels
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
        edge_data = _filter_components(edge_data, max_component_size, retire_groups=True)

    for u, v, sim, sup, lbl in zip(*edge_data):

        metadata = dict(weight=float(sim), bootstrap_support=float(sup))
        if lbl != "":
            metadata |= dict(edge_class=str(lbl))
        G.add_edge(u, v, **metadata)

    _assign_cluster_ids(G)

    return G


def _extract_graphdata(spectra: list[Spectrum], sim: np.ndarray, sup: np.ndarray) -> tuple[nx.Graph, EdgeData]:
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
    node_groups = np.arange(nr_of_nodes)   # each node starts in its own singleton cluster
    group_sizes = np.ones(nr_of_nodes)     # every cluster starts with size 1

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


def _assign_cluster_ids(graph: nx.Graph) -> None:
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    for cid, comp in enumerate(components):
        for node in comp:
            graph.nodes[node]["component"] = cid


def add_cluster_numbering(graph: nx.Graph):
    ordered_clusters = sorted(nx.connected_components(graph), key=lambda x: len(x), reverse=True)
    for i, cluster in enumerate(ordered_clusters):
        for node in cluster:
            graph.nodes[node]["mn_cluster_id"] = i
