import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np
import networkx as nx

from typing import  List
from argparse import Namespace

from utils.similarity_matrix import similarity_matrix
from utils.constants import *


ID_COUNTER = -1


def get_unique_id():
    global ID_COUNTER
    ID_COUNTER += 1
    return ID_COUNTER



def get_edges(matches: List[dict], cutoff=0.66) -> list[tuple[int, int]]:
    smiles = [m[KEY_SMILES] for m in matches] 
    similarity = similarity_matrix(smiles, smiles, "morgan", "dice")
    rows, cols = np.where(np.triu(similarity, k=1) > cutoff)
    edges = list(zip(rows, cols))
    return edges


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
    counts   = [_nr_of_unique_compounds(graph, c, KEY_MN_NODE_ID) for c in clusters]

    for c, nodes in zip(counts, clusters):
        for node in nodes:
            if max(counts) <= 2:
                graph.nodes[node][KEY_IS_TOP_CANDIDATE] = False
            else:
                graph.nodes[node][KEY_IS_TOP_CANDIDATE] = (c == max(counts))

            graph.nodes[node][KEY_ANN_MASS_DIVERSITY] = c


def _nr_of_unique_compounds(graph: nx.Graph, nodes: set, key: str) -> int:
    all_compounds = {graph.nodes[node][key] for node in nodes}
    return len(all_compounds)


def add_cluster_numbering(graph: nx.Graph):
    ordered_clusters = sorted(nx.connected_components(graph), key=lambda x: len(x), reverse=True)
    for cluster in ordered_clusters:
        i = get_unique_id()
        for node in cluster:
            graph.nodes[node][KEY_MN_CLUSTER_ID] = i
