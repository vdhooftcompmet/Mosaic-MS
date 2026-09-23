import numpy as np
import networkx as nx
from argparse import Namespace
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


ID_COUNTER = -1


def get_unique_id():
    global ID_COUNTER
    ID_COUNTER += 1
    return ID_COUNTER


def get_edges(matches: list[dict], cutoff=0.66) -> list[tuple[int, int]]:
    fingerprints = []
    for match in matches:
        fp = ExplicitBitVect(2048)
        fp.FromBase64(match["morgan_fingerprint"])
        fingerprints.append(fp)

    dice_matrix = np.zeros(shape=(len(matches), len(matches)))
    for i, fp in enumerate(fingerprints):
        dice_matrix[i, i+1:] = DataStructs.BulkDiceSimilarity(fp, fingerprints[i+1:])

    rows, cols = np.where(np.triu(dice_matrix, k=1) > cutoff)

    result = []
    for u, v in zip(rows, cols):
        if matches[u]["neutral_mass"] == matches[v]["neutral_mass"]:
            continue

        result.append((u, v))

    return result


def remove_self_similar_vals(edges):
    return [(u, v) for u, v in edges if u != v]


def remove_edges_with_same_value_for(edges, metadata, key):
    return [(u, v) for u, v in edges if metadata[u][key] != metadata[v][key]]
    

def remove_small_subgraphs(graph: nx.Graph, params: Namespace):
    clusters = [x for x in nx.connected_components(graph)]
    for nodes in clusters:
        if len(nodes) < params.min_annotation_size:
            graph.remove_nodes_from(nodes)


def add_top_candidate_annotation(graph: nx.Graph) -> None:
    clusters = [x for x in nx.connected_components(graph)]    
    counts   = [_nr_of_unique_compounds(graph, c, "mn_node_id") for c in clusters]

    for c, nodes in zip(counts, clusters):
        for node in nodes:
            if max(counts) <= 2:
                graph.nodes[node]["is_top_candidate"] = False
            else:
                graph.nodes[node]["is_top_candidate"] = (c == max(counts))

            graph.nodes[node]["ann_mass_diversity"] = c


def _nr_of_unique_compounds(graph: nx.Graph, nodes: set, key: str) -> int:
    all_compounds = {graph.nodes[node][key] for node in nodes}
    return len(all_compounds)


def add_cluster_numbering(graph: nx.Graph):
    ordered_clusters = sorted(nx.connected_components(graph), key=lambda x: len(x), reverse=True)
    for cluster in ordered_clusters:
        i = get_unique_id()
        for node in cluster:
            graph.nodes[node]["mn_cluster_id"] = i
