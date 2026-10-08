import argparse
from pathlib import Path

import yaml


def prepare_args(args: argparse.Namespace, default_config: Path) -> argparse.Namespace:
    """Applies YAML defaults, strips control attributes, and logs parameters."""

    args.defaults = (
        args.defaults if getattr(args, "defaults", None) else str(default_config)
    )
    if args.defaults and Path(args.defaults).exists():
        add_defaults(args, str(args.defaults))

    for key in ("defaults", "func", "command"):
        if hasattr(args, key):
            delattr(args, key)
    return args


def add_defaults(params, defaults_path) -> None:
    if defaults_path is not None:
        default_params = load_params(defaults_path)

        for k, v in dict(vars(default_params)).items():
            key_normalized = k.replace("-", "_")

            if not hasattr(params, key_normalized) or (
                getattr(params, key_normalized) is None
            ):
                setattr(params, key_normalized, v)


def load_params(path) -> argparse.Namespace:
    with open(path) as f:
        params = yaml.safe_load(f)
    return argparse.Namespace(**params)
