import argparse
from pathlib import Path

from src.utils.cli import prepare_args
from src.utils.configs import AnnotationConfig
from src.utils.cx import write_cx

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_ANNOTATION_CONFIG = REPO / "config" / "annotation.yaml"


def main(args: argparse.Namespace):
    args = prepare_args(args, DEFAULT_ANNOTATION_CONFIG)
    mn_config = AnnotationConfig(**dict(vars(args)))
    mn_config.display()

    from matchms.importing import load_from_mgf

    from src.annotation.cluster import build_spectra_graph
    from src.annotation.fragment import fragment_snapms
    from src.annotation.merge import merge_motifs

    motifs, motif_graph = merge_motifs(mn_config.models)
    spectra = list(load_from_mgf(mn_config.mgf))
    spectra_graph = build_spectra_graph(motifs, spectra)
    write_cx(spectra_graph, "xxxxxx.cx")

    fragment_snapms(mn_config, spectra_graph)


def build_parser():
    parser = argparse.ArgumentParser(description="motif fragment based SNAP-MS tool.")
    parser.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "-m",
        "--mgf",
        required=True,
        type=Path,
        help="Path to mgf file",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of models used to derive motifs",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=0.1,
        type=float,
        help="motif detection threshold",
    )
    parser.add_argument(
        "-g",
        "--graph",
        type=Path,
        default=None,
        help="Optional output graph .cx path (defaults to stdout)",
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
