import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.utils.constants import (
    KEY_ANN_MASS_DIVERSITY,
    KEY_ANNOTATION_QUALITY,
    KEY_ANNOTATION_QUALITY_SCORE,
    KEY_IS_TOP_CANDIDATE,
    KEY_MN_CLUSTER_ID,
    KEY_NEUTRAL_MASS,
    KEY_SMILES,
)
from src.utils.cx import read_annotations, read_cx
from src.utils.similarity_matrix import similarity_matrix


def calculate_quality_counts(mn: nx.Graph, annotations: dict[str, nx.Graph]) -> Counter:
    _assign_scores(mn, annotations)

    counts = Counter()
    for node in mn.nodes:
        quality_label = mn.nodes[node].get(KEY_ANNOTATION_QUALITY, "unavailable")
        counts[quality_label] += 1

    return counts


def _assign_scores(network: nx.Graph, annotations: dict[str, nx.Graph]):
    clusters = defaultdict(set)

    for node in network:
        cluster = str(network.nodes[node][KEY_MN_CLUSTER_ID])
        clusters[cluster].add(node)

    for cluster, nodes in clusters.items():
        annotation = annotations.get(str(cluster))

        for node in nodes:
            smiles = network.nodes[node][KEY_SMILES]
            label, score, maximum, diversity = __determine_node_quality(
                smiles, annotation
            )

            network.nodes[node][KEY_ANNOTATION_QUALITY_SCORE] = score
            network.nodes[node][KEY_ANNOTATION_QUALITY] = label
            network.nodes[node][KEY_ANN_MASS_DIVERSITY] = diversity

    network.graph.clear()


def __determine_node_quality(smiles, annotation: nx.Graph | None):
    if annotation is None:
        return "unavailable", 0.0, 0.0, 0

    top_candidates = []
    other_candidates = []

    maximum = 0.0
    diversity = [annotation.nodes[n][KEY_ANN_MASS_DIVERSITY] for n in annotation]
    diversity = [int(x) for x in diversity]
    diversity = max(diversity) if len(diversity) != 0 else 0

    for cluster in nx.connected_components(annotation):
        masses = {annotation.nodes[n][KEY_NEUTRAL_MASS] for n in cluster}
        if len(masses) <= 2:
            continue

        is_top_candidate = annotation.nodes[next(iter(cluster))][KEY_IS_TOP_CANDIDATE]
        annotation_smiles = [annotation.nodes[n][KEY_SMILES] for n in cluster]

        matrix = similarity_matrix(
            [smiles], annotation_smiles, fingerprint="morgan", matrix="dice"
        )
        similarity = np.max(matrix)
        maximum = max(maximum, similarity)

        is_quality_annotation = similarity >= 0.66

        if is_top_candidate:
            top_candidates.append(is_quality_annotation)
        else:
            other_candidates.append(is_quality_annotation)

    score = float(sum(top_candidates) / len(top_candidates)) if top_candidates else 0.0

    if len(top_candidates) == 0:
        label = "unavailable"
    elif score == 1.0:
        label = "good"
    elif score != 0.0:
        label = "mixed"
    elif any(other_candidates):
        label = "miss"
    else:
        label = "poor"

    return label, score, maximum, diversity


def main():
    parser = argparse.ArgumentParser(
        description="Compute annotation quality categories and output counts as a CSV."
    )
    parser.add_argument(
        "-m",
        "--mn-file",
        required=True,
        type=Path,
        help="Path to molecular network CX file",
    )
    parser.add_argument(
        "-a",
        "--annotation-folder",
        required=True,
        type=Path,
        help="Path to annotations folder",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (defaults to stdout)",
    )

    args = parser.parse_args()

    mn = read_cx(str(args.mn_file))
    annotations, files, cluster_ids = read_annotations(str(args.annotation_folder))
    annotation_dict = dict(zip(cluster_ids, annotations))

    counts = calculate_quality_counts(mn, annotation_dict)

    categories = ["good", "mixed", "poor", "miss", "unavailable"]
    df = pd.DataFrame([{cat: counts[cat] for cat in categories}])

    if args.output:
        df.to_csv(args.output, index=False)
    else:
        df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
