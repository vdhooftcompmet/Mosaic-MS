import argparse
from pathlib import Path

from src.utils.cli import prepare_args
from src.utils.configs import MergeConfig
from src.utils.cx import write_cx

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_ANNOTATION_CONFIG = REPO / "config" / "merge.yaml"


def main(args: argparse.Namespace):
    args = prepare_args(args, DEFAULT_ANNOTATION_CONFIG)
    mn_config = MergeConfig(**dict(vars(args)))
    mn_config.display()

    from matchms.exporting import save_as_mgf

    from src.merge.merge import merge_motifs

    motifs, G = merge_motifs(args.model_paths)
    save_as_mgf(motifs, args.out_motifs, file_mode="w")
    write_cx(G, args.out_graph)


def build_parser():
    parser = argparse.ArgumentParser(description="motif merge tool")
    parser.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "-m",
        "--motif-files",
        nargs="+",
        type=Path,
        help="Paths to Mass2Motifs (.mgf) or LDA model files (.lda), can be used interchangably",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=Path,
        default=None,
        help="Optional output graph .cx path (defaults to stdout)",
    )
    parser.add_argument(
        "-g",
        "--out-graph",
        type=Path,
        default=None,
        help="Optional output graph .cx path (defaults to stdout)",
    )
    parser.add_argument(
        "o",
        "--out-motifs",
        type=str,
        default=None,
        help="Destination directory for output.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
