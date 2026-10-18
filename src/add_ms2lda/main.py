import argparse
from pathlib import Path

from src.utils.cli import prepare_args
from src.utils.configs import AddMS2LDAConfig

REPO = Path(__file__).parent.parent.parent.resolve()

DEFAULT_ADD_MS2LDA_CONFIG = REPO / "config" / "add_ms2lda.yaml"


def main(args: argparse.Namespace):
    args = prepare_args(args, DEFAULT_ADD_MS2LDA_CONFIG)
    add_ms2lda_config = AddMS2LDAConfig(**dict(vars(args)))
    add_ms2lda_config.display()

    from src.add_ms2lda.add_ms2lda import add_ms2lda

    add_ms2lda(add_ms2lda_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--motifs",
        required=True,
        type=str,
        help="Path to motifs.",
    )
    parser.add_argument(
        "--graph",
        "-g",
        required=True,
        type=str,
        help="Path to target network CX graph file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Motif metadata overlap threshold.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
