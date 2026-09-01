import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import networkx as nx
import ndex2
import json
import contextlib

from utils.constants import *
from pathlib import Path
from tqdm import tqdm

import ndex2
import math  


def read_cx(file):
    # Load into the 'Nice' wrapper
    nice_cx = ndex2.create_nice_cx_from_file(file)
    
    # Use 'default' mode to ensure attributes are carried over correctly
    # and MultiDiGraph to prevent losing parallel edges
    graph = nice_cx.to_networkx(mode='default')
    
    # Optional: Fix node labels if they are stored in the 'n' attribute
    # but NetworkX used the internal ID as the node name
    if all(isinstance(n, int) for n in graph.nodes):
        mapping = {n: graph.nodes[n].get('n', n) for n in graph.nodes}
        graph = nx.relabel_nodes(graph, mapping)

    graph = graph.to_undirected()
    return graph


def write_cx(graph: nx.Graph, file_path: str, style_example: str | None = None):
    nice_cx = ndex2.NiceCXNetwork()
    
    # 1. Map NetworkX nodes to CX internal IDs
    # This is the most common reason for 'missing edges'
    node_name_to_id = {}
    for node in graph.nodes:
        # Create node and store its internal ID
        node_id = nice_cx.create_node(node_name=str(node))
        node_name_to_id[node] = node_id

        # Add node attributes with type detection
        for k, v in graph.nodes[node].items():
            nice_cx.set_node_attribute(node_id, k, v, type=infer_cx_type(v))

    # 2. Add Edges using the ID mapping
    for source, target, metadata in graph.edges(data=True):
        s_id = node_name_to_id[source]
        t_id = node_name_to_id[target]
        
        edge_id = nice_cx.create_edge(edge_source=s_id, edge_target=t_id)

        # Add edge attributes
        for k, v in metadata.items():
            nice_cx.set_edge_attribute(edge_id, k, v, type=infer_cx_type(v))

    # 3. Handle Styles & Metadata
    if style_example:
        style_cx = ndex2.create_nice_cx_from_file(str(style_example))
        nice_cx.apply_style_from_network(style_cx)

    # Use the built-in uploader or manual dump
    with open(file_path, 'w') as f:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            cx_data = nice_cx.to_cx()
        json.dump(cx_data, f)

def infer_cx_type(val):
    if isinstance(val, bool): return "boolean"
    if isinstance(val, int): return "integer"
    if isinstance(val, float): return "double"
    return "string"


def read_annotations(folder: str | Path):
    annotations, files, cluster_ids = [], [], []
    for annotation_file in tqdm(Path(folder).glob("*.cx")):
    
        try:
            annotation = read_cx(annotation_file)
        except Exception:
            print(f"ERROR reading file {annotation_file}")
            continue

        mn_cluster_id = annotation_file.stem.split("-")[1]
        cluster_ids += [mn_cluster_id]
        annotations += [annotation]
        files       += [annotation_file]

    return annotations, files, cluster_ids
