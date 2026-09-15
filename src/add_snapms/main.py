import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


import networkx as nx 

from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from setup.paths import MN_STYLE_FILE
from utils.constants import KEY_MN_CLUSTER_ID, KEY_ANN_MASS_DIVERSITY, KEY_IS_ANNOTATED, KEY_IS_TOP_CANDIDATE, KEY_MN_NODE_ID
from utils.cx import write_cx, read_cx, read_annotations
from setup.paths import ANNOTATION_STYLE_FILE, MN_STYLE_FILE

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import networkx as nx

from collections import defaultdict

from utils.similarity_matrix import similarity_matrix
from utils.constants import *
from utils.cx import write_cx

from setup.paths import ANNOTATION_STYLE_FILE, MN_STYLE_FILE

from utils.cx import write_cx

from utils.cx import write_cx, read_cx, read_annotations

from setup.paths import ANNOTATION_STYLE_FILE, MN_STYLE_FILE


def main(params):
    mn = read_cx(str(params.graph))

    annotations, files, cluster_ids = read_annotations(params.snapms)

    cluster_ids = [str(x) for x in cluster_ids]
    annotation_dict = dict(zip(cluster_ids, annotations))
    add_snapms_data(mn, annotation_dict)

    annotation_dict = dict(zip(cluster_ids, annotations))
    set_smiles_flag(mn, annotation_dict)

    write_cx(mn, str(params.graph), MN_STYLE_FILE)
    for file, annotation in zip(files, annotations):
        write_cx(annotation, file, ANNOTATION_STYLE_FILE)


def add_snapms_data(mn: nx.Graph, annotations: dict[str | int, nx.Graph]) -> None:
    clusters = defaultdict(set)

    for node in mn:
        cluster = str( mn.nodes[node][KEY_MN_CLUSTER_ID] )
        clusters[cluster].add(node)

    for cluster, nodes in clusters.items():
        for node in nodes:
            annotation = annotations.get(cluster, [])

            if len(annotation) == 0:
                mn.nodes[node][KEY_ANN_MASS_DIVERSITY] = 0
                continue

            max_diversity = max([annotation.nodes[n][KEY_ANN_MASS_DIVERSITY] for n in annotation])
            mn.nodes[node][KEY_ANN_MASS_DIVERSITY] = max_diversity


    for node in mn:
        mn.nodes[node][KEY_IS_ANNOTATED] = False

    for cluster_id, graph in annotations.items():
        is_top_candidate = any([graph.nodes[n][KEY_IS_TOP_CANDIDATE] for n in graph])

        for node in clusters[cluster_id]:
            mn.nodes[node][KEY_IS_ANNOTATED] = is_top_candidate


def set_smiles_flag(network: nx.Graph, annotations: dict[int | nx.Graph]) -> None:
    
    clusters = defaultdict(list)

    for node in network:
        cluster = network.nodes[node][KEY_MN_CLUSTER_ID]
        clusters[cluster].append( node )

    for cluster_index, annotation in annotations.items():
        cluster = clusters.get(cluster_index, [])

        valid_cluster_nodes     = [n for n in cluster    if network.nodes[n].get(KEY_SMILES)]
        valid_annotation_nodes  = [n for n in annotation if annotation.nodes[n].get(KEY_SMILES)]

        cluster_smiles      = [network.nodes[n].get(KEY_SMILES)    for n in valid_cluster_nodes]
        annotation_smiles   = [annotation.nodes[n].get(KEY_SMILES) for n in valid_annotation_nodes]

        if (not cluster_smiles) or (not annotation_smiles):
            continue

        similarity = similarity_matrix(cluster_smiles, annotation_smiles, "morgan", "dice")

        for i, node in enumerate(valid_annotation_nodes):
            matches = similarity[:, i]
            is_match = bool( np.max(matches) >= 0.66 )
            annotation.nodes[node][KEY_HAS_SMILES_MATCH] = is_match
