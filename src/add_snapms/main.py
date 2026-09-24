import networkx as nx 
from collections import defaultdict
from setup.paths import MN_STYLE_FILE
from utils.constants import KEY_MN_CLUSTER_ID, KEY_ANN_MASS_DIVERSITY, KEY_IS_ANNOTATED, KEY_IS_TOP_CANDIDATE, KEY_MN_NODE_ID
from utils.cx import write_cx, read_cx, read_annotations
from setup.paths import ANNOTATION_STYLE_FILE, MN_STYLE_FILE
import numpy as np
from utils.similarity_matrix import similarity_matrix
from utils.constants import *
from setup.paths import ANNOTATION_STYLE_FILE, MN_STYLE_FILE


def main(params):
    mn = read_cx(str(params.graph))

    annotations, files, cluster_ids = read_annotations(params.snapms)

    cluster_ids = [str(x) for x in cluster_ids]
    annotation_dict = dict(zip(cluster_ids, annotations))
    add_snapms_data(mn, annotation_dict)

    write_cx(mn, str(params.graph), MN_STYLE_FILE)
    for file, annotation in zip(files, annotations):
        write_cx(annotation, file, ANNOTATION_STYLE_FILE)


def add_snapms_data(mn: nx.Graph, annotations: dict) -> None:
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

