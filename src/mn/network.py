import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import networkx as nx
import numpy as np

from matchms import Spectrum
from argparse import Namespace

from utils.constants import *


def run_networking(spectra, similarity, support, similarity_type, network_type, params):

    match network_type:
        case "base":
            fn = base_graph
        case "threshold":
            fn = threshold_graph
        case "rescued":
            fn = rescued_graph
        case _: 
            raise ValueError(f"{network_type = }")

    similarity = np.nan_to_num(similarity, nan=0.0)
    support = np.nan_to_num(support, nan=0.0)
    graph = fn(similarity, support, spectra, similarity_type, params)

    return graph


def base_graph(
        mean_similarities: np.ndarray, 
        mean_edge_support: np.ndarray, 
        spectra: list[Spectrum], 
        similarity_type: str,
        params: Namespace
) -> nx.Graph:
    

    edges = np.ones_like(mean_similarities)
    edges[mean_similarities < float(params.similarity_threshold)] = 0
    np.fill_diagonal(edges , 0)  

    graph = build_nodes(spectra)

    rows, cols = np.nonzero(edges)
    if len(rows) == 0 or len(cols) == 0:
        print("WARNING: no edges could be made")
        return graph

    weights  = mean_similarities[rows, cols]
    
    mask = _filter_components(rows, cols, weights, params.max_component_size)

    rows = rows[mask]
    cols = cols[mask]

    if len(rows) == 0 or len(cols) == 0:
        print("WARNING: no edges left after filtering")
        return graph
    
    weights  = mean_similarities[rows, cols]
    supports = mean_edge_support[rows, cols]

    for row, col, weight, support in zip(rows, cols, weights, supports):
        if params.similarity_threshold is not None:
            edge_class = "core" if weight >= params.similarity_threshold else "rescued"
            graph.add_edge(row, col, edge_class=edge_class, weight=weight, bootstrap_support=support)
        else:
            graph.add_edge(row, col, weight=weight, bootstrap_support=support)

    add_cluster_numbering(graph)

    graph.graph.clear()
    return graph
    

def threshold_graph(
        mean_similarities: np.ndarray, 
        mean_edge_support: np.ndarray, 
        spectra: list[Spectrum], 
        similarity_type: str,
        params: Namespace
) -> nx.Graph:
    

    edges = np.ones_like(mean_similarities)
    edges[mean_similarities < float(params.similarity_threshold)] = 0
    edges[mean_edge_support < float(params.support_threshold)] = 0
    np.fill_diagonal(edges , 0)  


    graph = build_nodes(spectra)

    rows, cols = np.nonzero(edges)
    if len(rows) == 0 or len(cols) == 0:
        print("WARNING: no edges could be made")
        return graph
    
    weights  = mean_similarities[rows, cols]
    
    mask = _filter_components(rows, cols, weights, params.max_component_size)

    rows = rows[mask]
    cols = cols[mask]

    if len(rows) == 0 or len(cols) == 0:
        print("WARNING: no edges left after filtering")
        return graph
    
    weights  = mean_similarities[rows, cols]
    supports = mean_edge_support[rows, cols]

    for row, col, weight, support in zip(rows, cols, weights, supports):
        if params.similarity_threshold is not None:
            edge_class = "core" if weight >= params.similarity_threshold else "rescued"
            graph.add_edge(row, col, edge_class=edge_class, weight=weight, bootstrap_support=support)
        else:
            graph.add_edge(row, col, weight=weight, bootstrap_support=support)

    add_cluster_numbering(graph)

    graph.graph.clear()
    return graph


def rescued_graph(
        mean_similarities: np.ndarray, 
        mean_edge_support: np.ndarray, 
        spectra: list[Spectrum], 
        similarity_type: str,
        params: Namespace
) -> nx.Graph:

    edges = np.ones_like(mean_similarities)
    edges[mean_similarities < float(params.rescue_similarity_threshold)] = 0
    edges[mean_edge_support < float(params.support_threshold)] = 0
    np.fill_diagonal(edges , 0)  
    edges = edges.astype(bool) 

    graph = build_nodes(spectra)

    rows, cols = np.nonzero(edges)
    if len(rows) == 0 or len(cols) == 0:
        print("WARNING: no edges could be made")
        return graph

    weights  = mean_similarities[rows, cols]
    
    mask = _filter_components(rows, cols, weights, params.max_component_size)

    rows = rows[mask]
    cols = cols[mask]

    if len(rows) == 0 or len(cols) == 0:
        print("WARNING: no edges left after filtering")
        return graph
    
    weights  = mean_similarities[rows, cols]
    supports = mean_edge_support[rows, cols]

    for row, col, weight, support in zip(rows, cols, weights, supports):
        if params.similarity_threshold is not None:
            edge_class = "core" if weight >= params.similarity_threshold else "rescued"
            graph.add_edge(row, col, edge_class=edge_class, weight=weight, bootstrap_support=support)
        else:
            graph.add_edge(row, col, weight=weight, bootstrap_support=support)

    add_cluster_numbering(graph)


    graph.graph.clear()
    return graph


def build_nodes(spectra: list[Spectrum]) -> nx.Graph:
    graph = nx.Graph()

    for index, spectrum in enumerate(spectra):
        metadata = {k: str(v) for k, v in spectrum.to_dict().items()}

        if not metadata:
            print(f"{spectrum = }")
        assert metadata

        graph.add_node(index, **metadata)

    return graph


def add_cluster_numbering(graph: nx.Graph):
    ordered_clusters = sorted(nx.connected_components(graph), key=lambda x: len(x), reverse=True)
    for i, cluster in enumerate(ordered_clusters):
        for node in cluster:
            graph.nodes[node][KEY_MN_CLUSTER_ID] = i


def _filter_components(u_nodes: np.array, v_nodes: np.array, similarity_array: np.array, max_component_size: int, retire_groups: bool = False) -> np.array:

    retired_groups = set()
    
    nr_of_nodes = max(np.max(u_nodes), np.max(v_nodes)) + 1
    node_groups = np.array(range(nr_of_nodes))  # lookup
    group_sizes = np.ones(nr_of_nodes) 

    similarity_array = similarity_array.copy()  # make a copy to modify

    indices = np.argsort(similarity_array)[::-1]  # indices of numbers from high to low

    mask = np.zeros_like(similarity_array)

    for i in indices:
        strength = similarity_array[i]
        u, v = u_nodes[i], v_nodes[i]

        if strength == 0:  # encountering a strength of 0 means we're not going to add any more edges on the remaining data, so we can end the process (remember we sorted them by size)
            break

        u_group = node_groups[u]  # look up the group of u
        v_group = node_groups[v]  # look up the group of v

        if retire_groups and any(g in retired_groups for g in [u_group, v_group]):  # if we turned on this setting, we don't touch the retired groups (matches behaviour of original breakup implementation)
            retired_groups.add(u_group)
            retired_groups.add(v_group)  # we need to make sure BOTH groups are retired after a failed connection
            continue

        if u_group == v_group:  # if they're already in the same cluster, the cluster won't grow in size
            mask[i] = 1
            continue

        u_group_size = group_sizes[u_group]  # get the size of group u
        v_group_size = group_sizes[v_group]  # get the size of group v

        group_sum = u_group_size + v_group_size
        if group_sum > max_component_size:  # adding these clusters would exceed the max size
            retired_groups.add(u_group)
            retired_groups.add(v_group)
            continue

        # if we get here, we're allowed to add the clusters
        mask[i] = 1

        # we need to update our group administration
        dominant_group, purged_group = sorted((u_group, v_group))  # determine which group will take over the members of the other (lowest group nr is dominant)

        node_groups[node_groups == purged_group] = dominant_group
        group_sizes[dominant_group] = group_sum
        group_sizes[purged_group] = 0


    return mask.astype(bool)
