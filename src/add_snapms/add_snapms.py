from collections import defaultdict

from src.utils.configs import AddSNAPMSConfig
from src.utils.cx import read_annotations, read_cx, write_cx
from src.utils.paths import ANNOTATION_STYLE_FILE, MN_STYLE_FILE
from src.utils.progress_bar import track


def add_snapms_data(config: AddSNAPMSConfig) -> None:
    mn = read_cx(str(config.graph))

    annotations_list, files, cluster_ids = read_annotations(config.snapms)

    cluster_ids = [str(x) for x in cluster_ids]
    annotations = dict(zip(cluster_ids, annotations_list))

    clusters = defaultdict(set)

    for node in mn:
        cluster = str(mn.nodes[node]["mn_cluster_id"])
        clusters[cluster].add(node)

    for cluster, nodes in track(clusters.items(), description="modifying mn graph..."):
        for node in nodes:
            annotation = annotations.get(cluster, [])

            if len(annotation) == 0:
                mn.nodes[node]["ann_mass_diversity"] = 0
                continue

            max_diversity = max(
                [annotation.nodes[n]["ann_mass_diversity"] for n in annotation]
            )
            mn.nodes[node]["ann_mass_diversity"] = max_diversity

    for node in mn:
        mn.nodes[node]["is_annotated"] = False

    for cluster_id, graph in track(
        annotations.items(), description="modifying snapms graphs..."
    ):
        is_top_candidate = any([graph.nodes[n]["is_top_candidate"] for n in graph])

        for node in clusters[cluster_id]:
            mn.nodes[node]["is_annotated"] = is_top_candidate

    write_cx(mn, str(config.graph), MN_STYLE_FILE)
    for file, annotation in zip(files, annotations_list):
        write_cx(annotation, file, ANNOTATION_STYLE_FILE)
