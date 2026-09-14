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


def main(params):
    mn = read_cx(params.mn_file)

    annotations, files, cluster_ids = read_annotations(params.annotation_folder)

    cluster_ids = [str(x) for x in cluster_ids]
    annotation_dict = dict(zip(cluster_ids, annotations))
    add_snapms_data(mn, annotation_dict)

    write_cx(mn, params.mn_file, MN_STYLE_FILE)
    for file, annotation in zip(files, annotations):
        write_cx(annotation, file, ANNOTATION_STYLE_FILE)

    write_cx(mn, params.mn_file)


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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--mn-file", required=True)
    parser.add_argument("--annotation-folder", required=True)

    params = parser.parse_args()
    main(params)
    

