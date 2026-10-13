import argparse
from pathlib import Path

from src.utils.cli import prepare_args
from src.utils.configs import SNAPMSConfig

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_SNAPMS_CONFIG = REPO / "config" / "snapms.yaml"


def main(args: argparse.Namespace) -> None:
    args = prepare_args(args, DEFAULT_SNAPMS_CONFIG)
    config = SNAPMSConfig(**dict(vars(args)))
    config.display()

    from src.snapms.snapms import snapms

    snapms(config)


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
