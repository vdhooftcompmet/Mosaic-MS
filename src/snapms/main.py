import shutil
from pathlib import Path

import networkx as nx

from src.snapms.snapms import (
    build_molecular_families,
    find_db_matches,
    group_nodes,
    import_atlas,
)
from src.utils.configs import SNAPMSConfig
from src.utils.cx import read_cx, write_cx
from src.utils.folders import prepare_directory
from src.utils.paths import ANNOTATION_STYLE_FILE
from src.utils.progress_bar import track


def main(config: SNAPMSConfig) -> None:
    atlas_con = import_atlas(config)

    file = Path(config.graph)

    mn = read_cx(str(file))

    annotation_folder = Path(config.result_folder)
    if annotation_folder.is_dir():
        shutil.rmtree(annotation_folder)
    if not Path(annotation_folder).exists():
        prepare_directory(annotation_folder)

    groups = group_nodes(mn)

    for identifier, nodes in track(groups.items(), description="running snap-ms..."):
        if len(nodes) < config.min_cluster_size:
            continue
        if len(nodes) > config.max_cluster_size:
            continue

        matches = find_db_matches(mn, nodes, config, atlas_con)
        G = build_molecular_families(matches, config)
        if G is None:
            continue

        # AFTER
        values = {node: identifier for node in G.nodes}
        nx.set_node_attributes(G=G, values=values, name="mn_cluster_id")

        save_path = str(annotation_folder / f"graph-{identifier}.cx")
        write_cx(G, save_path, ANNOTATION_STYLE_FILE)
