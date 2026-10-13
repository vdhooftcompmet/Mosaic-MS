import argparse
from pathlib import Path

from src.utils.cli import prepare_args
from src.utils.configs import AddMAGConfig

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_ADD_MAG_CONFIG = REPO / "config" / "add_mag.yaml"


def main(args: argparse.Namespace) -> None:
    args = prepare_args(args, DEFAULT_ADD_MAG_CONFIG)
    config = AddMAGConfig(**dict(vars(args)))
    config.display()

    assert Path(config.library).exists(), "missing library"

    from src.ms2lda.mag import add_mag

    add_mag(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--motifs",
        required=True,
        type=str,
        help="Path to extracted Mass2Motifs MGF file.",
    )
    parser.add_argument(
        "--spec2vec-model-path",
        type=str,
        default=None,
        help="Path to Spec2Vec model file.",
    )
    parser.add_argument(
        "--library",
        type=str,
        default=None,
        help="Path to reference library MGF file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity score threshold.",
    )
    parser.add_argument(
        "--cluster-delta",
        type=float,
        default=None,
        help="Cluster delta threshold.",
    )
    parser.add_argument(
        "--criterium",
        type=str,
        default=None,
        help="Selection criterium metric.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
