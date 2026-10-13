import argparse
from pathlib import Path

from src.utils.cli import prepare_args
from src.utils.configs import AddSNAPMSConfig

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_ADD_SNAPMS_CONFIG = REPO / "config" / "add_snapms.yaml"


def main(args: argparse.Namespace) -> None:
    args = prepare_args(args, DEFAULT_ADD_SNAPMS_CONFIG)
    config = AddSNAPMSConfig(**dict(vars(args)))
    config.display()

    from src.add_snapms.add_snapms import add_snapms_data

    add_snapms_data(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--graph",
        "-g",
        required=True,
        type=str,
        help="Path to target network CX graph file.",
    )
    parser.add_argument(
        "--snapms",
        "-a",
        required=True,
        type=str,
        help="Path to SNAP-MS annotation results directory containing output files.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
