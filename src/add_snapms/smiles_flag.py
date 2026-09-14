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
    mn = read_cx(params.mn_file)

    annotations, files, cluster_ids = read_annotations(params.annotation_folder)

    annotation_dict = dict(zip(cluster_ids, annotations))
    set_smiles_flag(mn, annotation_dict)

    write_cx(mn, params.mn_file, MN_STYLE_FILE)
    for file, annotation in zip(files, annotations):
        write_cx(annotation, file, ANNOTATION_STYLE_FILE)


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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--mn-file", required=True)
    parser.add_argument("--annotation-folder", required=True)
    
    params = parser.parse_args()
    main(params)
    