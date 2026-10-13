import argparse
from pathlib import Path

from src.utils.cli import prepare_args
from src.utils.configs import MNConfig

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_MN_CONFIG = REPO / "config" / "mn.yaml"


def main(args: argparse.Namespace) -> None:
    args = prepare_args(args, DEFAULT_MN_CONFIG)
    config = MNConfig(**dict(vars(args)))
    config.display()

    from src.mn.mn import mn

    mn(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI tool for strata-ms workflows.")
    parser.add_argument(
        "--defaults",
        type=str,
        default=None,
        help="Path to a YAML configuration file containing default parameters.",
    )
    parser.add_argument(
        "--mgf",
        "-m",
        type=str,
        default=None,
        help="File path pointing towards the mass spectra (MGF) to be analyzed.",
    )
    parser.add_argument(
        "--similarity-type",
        choices=["cos", "modcos", "spec2vec", "ms2deepscore"],
        default=None,
        help="Similarity metric used for networking.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Minimum similarity threshold required to connect two nodes with an edge.",
    )
    parser.add_argument(
        "--flash-tolerance",
        type=float,
        default=None,
        help="Mass tolerance error margin (in Da).",
    )
    parser.add_argument(
        "--binning-decimals",
        type=int,
        default=None,
        help="Precision used to bin spectrum m/z values.",
    )
    parser.add_argument(
        "--max-component-size",
        type=int,
        default=None,
        help="Maximum allowable cluster size in the network.",
    )
    parser.add_argument(
        "--B", type=int, default=None, help="Bootstrapping iterations per spectrum."
    )
    parser.add_argument("--k", type=int, default=None, help="Top-k nearest neighbors.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--ms2deepscore-model-path",
        type=str,
        default=None,
        help="MS2DeepScore model path.",
    )
    parser.add_argument(
        "--spec2vec-model-path", type=str, default=None, help="Spec2Vec model path."
    )
    parser.add_argument(
        "--use-average-similarity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use average similarity from bootstrapping.",
    )
    parser.add_argument(
        "--cache-similarity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache and reuse similarity calculations.",
    )
    parser.add_argument(
        "--base-graph-path",
        type=str,
        default=None,
        help="Path for base network graph (.cx).",
    )
    parser.add_argument(
        "--threshold-graph-path",
        type=str,
        default=None,
        help="Path for threshold graph (.cx).",
    )
    parser.add_argument(
        "--rescued-graph-path",
        type=str,
        default=None,
        help="Path for rescued graph (.cx).",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
