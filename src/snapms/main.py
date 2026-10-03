from collections import defaultdict
from pathlib import Path

import networkx as nx

from src.snapms.snapms import (
    add_cluster_numbering,
    add_top_candidate_annotation,
    compute_adduct_matches,
    get_edges,
    import_atlas,
    merge_duplicates,
    remove_edges_with_same_value_for,
    remove_self_similar_vals,
    remove_small_subgraphs,
)
from src.utils.configs import SNAPMSConfig
from src.utils.cx import read_cx, write_cx
from src.utils.folders import prepare_directory
from src.utils.paths import ANNOTATION_STYLE_FILE
from src.utils.progress_bar import track


def main(config: SNAPMSConfig) -> None:
    atlas_df = import_atlas(config)

    file = Path(config.graph)

    mn = read_cx(str(file))

    annotation_folder = Path(config.result_folder)
    if not Path(annotation_folder).exists():
        prepare_directory(annotation_folder)

    clusters: dict[int, list[int]] = defaultdict(list)
    for node in mn:
        mn_cluster_id = mn.nodes[node]["mn_cluster_id"]
        clusters[mn_cluster_id].append(node)

    for mn_cluster_id, nodes in track(
        clusters.items(), description="running snap-ms..."
    ):
        if len(nodes) < config.min_cluster_size:
            continue
        if len(nodes) > config.max_cluster_size:
            continue

        matches = compute_adduct_matches(mn, nodes, config, atlas_df)
        matches = merge_duplicates(
            matches
        )  # nodes with the same or very similar masses lead to multiple copies of compounds, here we merge them into one

        edges = get_edges(matches, cutoff=config.cutoff)
        edges = remove_self_similar_vals(
            edges
        )  # makes sure nodes aren't connected to themselves
        edges = remove_edges_with_same_value_for(
            edges, matches, "mn_node_id"
        )  # snapms logic dictates compounds from the same origin node cannot connect to each other

        graph = nx.Graph()
        graph.add_nodes_from((i, match) for i, match in enumerate(matches))
        graph.add_edges_from(edges)

        remove_small_subgraphs(graph, config)  # small families are likely irrelevant

        if len(graph) == 0:  # empty graphs are not saved for ease of user investigation
            continue

        add_cluster_numbering(graph)
        add_top_candidate_annotation(graph)

        values = {node: mn_cluster_id for node in graph.nodes}
        nx.set_node_attributes(G=graph, values=values, name="mn_cluster_id")

        save_path = str(annotation_folder / f"graph-{mn_cluster_id}.cx")
        write_cx(graph, save_path, ANNOTATION_STYLE_FILE)
