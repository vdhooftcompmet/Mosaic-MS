import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from utils.folders import prepare_directory
from utils.constants import *
from utils.cx import read_cx, write_cx
from setup.paths import ANNOTATION_STYLE_FILE
from snapms.masses import import_atlas, compute_adduct_matches, merge_duplicates
from snapms.network import get_edges, remove_edges_with_same_value_for, remove_self_similar_vals, remove_small_subgraphs, add_cluster_numbering, add_top_candidate_annotation
from utils.configs import SNAPMSConfig


def main(config: SNAPMSConfig):
    atlas_df = import_atlas(config)

    file = Path(config.graph)

    try:
        mn = read_cx(str(file))
    except (ValueError, OSError, FileExistsError, FileNotFoundError) as e:
        print(f"WARNING: failed to read file {file}; error: {e}")
        return

    annotation_folder = Path(config.result_folder)
    if not Path(annotation_folder).exists():
        prepare_directory(annotation_folder)
    
    clusters: dict[int, list[int]] = defaultdict(list)
    for node in mn:
        mn_cluster_id = mn.nodes[node][KEY_MN_CLUSTER_ID]
        clusters[mn_cluster_id].append(node)

    for mn_cluster_id, nodes in tqdm(clusters.items()):
        if len(nodes) < config.min_cluster_size:
            continue
        if len(nodes) > config.max_cluster_size:
            continue

        matches = compute_adduct_matches(mn, nodes, config, atlas_df)
        matches = merge_duplicates(matches)  # nodes with the same or very similar masses lead to multiple copies of compounds, here we merge them into one

        edges = get_edges(matches, cutoff=config.cutoff)
        edges = remove_self_similar_vals(edges)  # makes sure nodes aren't connected to themselves
        edges = remove_edges_with_same_value_for(edges, matches, KEY_MN_NODE_ID)  # snapms logic dictates compounds from the same origin node cannot connect to each other

        graph  = nx.Graph()
        graph.add_nodes_from((i, match) for i, match in enumerate(matches))
        graph.add_edges_from(edges)

        remove_small_subgraphs(graph, config)  # small families are likely irrelevant

        if len(graph) == 0:  # empty graphs are not saved for ease of user investigation
            continue

        add_cluster_numbering(graph)
        add_top_candidate_annotation(graph)
        nx.set_node_attributes(graph, mn_cluster_id, KEY_MN_CLUSTER_ID)

        save_path = annotation_folder / f"graph-{mn_cluster_id}.cx"
        write_cx(graph, save_path, ANNOTATION_STYLE_FILE)

