import argparse
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.utils.cx import read_annotations, read_cx
from src.utils.similarity_matrix import similarity_matrix


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

    counts = score_annotation(mn, annotation_dict)

    categories = ["good", "poor", "miss", "unavailable"]
    df = pd.DataFrame([{cat: counts.get(cat, 0) for cat in categories}])

    if args.output:
        df.to_csv(args.output, index=False)
    else:
        df.to_csv(sys.stdout, index=False)


def score_annotation(
    network: nx.Graph, annotations: dict[str, nx.Graph]
) -> dict[str, int]:

    counts = defaultdict(int)
    clusters = defaultdict(set)

    for node in network:
        cluster = str(network.nodes[node]["mn_cluster_id"])
        clusters[cluster].add(node)

    for cluster, nodes in clusters.items():
        annotation = annotations.get(str(cluster))

        for node in nodes:
            smiles = network.nodes[node]["smiles"]
            score = calculate_score(smiles, annotation)

            for k, v in score.items():
                counts[k] += v

    return counts


def calculate_score(reference_smiles: str, annotation: nx.Graph | None) -> dict:
    if annotation is None:
        return {"unavailable": 1}

    top_candidates, other_candidates = [], []

    for cluster in nx.connected_components(annotation):
        first_node = next(iter(cluster))
        is_top_candidate = annotation.nodes[first_node]["is_top_candidate"]

        smiles = [annotation.nodes[n]["smiles"] for n in cluster]
        matrix = similarity_matrix(
            [reference_smiles], smiles, fingerprint="morgan", matrix="dice"
        )
        similarity = np.max(matrix)
        is_quality_annotation = similarity >= 0.66

        if is_top_candidate:
            top_candidates.append(is_quality_annotation)
        else:
            other_candidates.append(is_quality_annotation)

    if len(top_candidates) == 0:
        good = 0
    else:  # formula captures edge cases where there are multiple top_candidates
        good = sum(top_candidates) / len(top_candidates)

    if any(top_candidates + other_candidates):
        miss = 1 - good  # compensates for partial good score
        poor = 0
    else:
        miss = 0
        poor = 1

    assert sum([good, poor, miss]) == 1

    return {"good": good, "poor": poor, "miss": miss}


if __name__ == "__main__":
    main()
