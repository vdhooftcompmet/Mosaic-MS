import argparse
from collections import defaultdict
from pathlib import Path

import networkx as nx

from src.utils.cli import prepare_args
from src.utils.configs import AddSNAPMSConfig
from src.utils.cx import read_annotations, read_cx, write_cx
from src.utils.paths import ANNOTATION_STYLE_FILE, MN_STYLE_FILE
from src.utils.progress_bar import track

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_ADD_SNAPMS_CONFIG = REPO / "config" / "add_snapms.yaml"


def main(args: argparse.Namespace) -> None:
    args = prepare_args(args, DEFAULT_ADD_SNAPMS_CONFIG)
    config = AddSNAPMSConfig(**dict(vars(args)))
    config.display()

    mn = read_cx(str(config.graph))

    annotations, files, cluster_ids = read_annotations(config.snapms)

    cluster_ids = [str(x) for x in cluster_ids]
    annotation_dict = dict(zip(cluster_ids, annotations))
    add_snapms_data(mn, annotation_dict)

    write_cx(mn, str(config.graph), MN_STYLE_FILE)
    for file, annotation in zip(files, annotations):
        write_cx(annotation, file, ANNOTATION_STYLE_FILE)


def add_snapms_data(mn: nx.Graph, annotations: dict) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--graph",
        "-g",
        required=True,
        type=str,
        help="Path to target network CX graph file.",
    )
    parser.add_argument(
        "--snapms",
        "-a",
        required=True,
        type=str,
        help="Path to SNAP-MS annotation results directory containing output files.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
