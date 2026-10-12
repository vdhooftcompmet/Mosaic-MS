import argparse
import shutil
from pathlib import Path

import networkx as nx

from src.utils.cli import prepare_args
from src.utils.configs import SNAPMSConfig
from src.utils.cx import read_cx, write_cx
from src.utils.folders import prepare_directory
from src.utils.paths import ANNOTATION_STYLE_FILE
from src.utils.progress_bar import track

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_SNAPMS_CONFIG = REPO / "config" / "snapms.yaml"


def main(args: argparse.Namespace) -> None:
    args = prepare_args(args, DEFAULT_SNAPMS_CONFIG)
    config = SNAPMSConfig(**dict(vars(args)))
    config.display()

    from src.snapms.snapms import (
        build_molecular_families,
        find_db_matches,
        group_nodes,
        import_atlas,
    )

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

        values = {node: identifier for node in G.nodes}
        nx.set_node_attributes(G=G, values=values, name="mn_cluster_id")

        save_path = str(annotation_folder / f"graph-{identifier}.cx")
        write_cx(G, save_path, ANNOTATION_STYLE_FILE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--graph",
        "-g",
        type=str,
        default=None,
        help="Target molecular network CX graph.",
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        default=None,
        help="Destination directory for output.",
    )
    parser.add_argument(
        "--reference-db",
        type=str,
        default=None,
        help="Reference structure database path.",
    )
    parser.add_argument(
        "--ppm-error", type=float, default=None, help="Mass tolerance error in PPM."
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Minimum network cluster size.",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=None,
        help="Maximum network cluster size.",
    )
    parser.add_argument(
        "--min-annotation-size",
        type=int,
        default=None,
        help="Minimum annotation cluster size.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Tanimoto similarity cutoff threshold.",
    )
    parser.add_argument(
        "--adduct-list", nargs="+", default=None, help="List of candidate adduct forms."
    )
    parser.add_argument(
        "--detect-adduct",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatically detect adduct annotations directly from graph properties.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
