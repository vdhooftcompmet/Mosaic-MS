import networkx as nx 
from collections import defaultdict
from utils.cx import write_cx, read_cx, read_annotations
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
        cluster = str( mn.nodes[node]["mn_cluster_id"] )
        clusters[cluster].add(node)

    for cluster, nodes in clusters.items():
        for node in nodes:
            annotation = annotations.get(cluster, [])

            if len(annotation) == 0:
                mn.nodes[node]["ann_mass_diversity"] = 0
                continue

            max_diversity = max([annotation.nodes[n]["ann_mass_diversity"] for n in annotation])
            mn.nodes[node]["ann_mass_diversity"] = max_diversity


    for node in mn:
        mn.nodes[node]["is_annotated"] = False

    for cluster_id, graph in annotations.items():
        is_top_candidate = any([graph.nodes[n]["is_top_candidate"] for n in graph])

        for node in clusters[cluster_id]:
            mn.nodes[node]["is_annotated"] = is_top_candidate

