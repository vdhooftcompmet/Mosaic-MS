from argparse import Namespace
from pathlib import Path
import yaml
from utils.folders import load_params


def stringify_args(args: Namespace | None, ignore: list[str] | None = None) -> str:
    if ignore is None:
        ignore = []

    if not args:
        return ""

    params = vars(args)
    if not params:
        return ""

    params = {k: v for k, v in params.items() if v is not None}
    params = {k: v for k, v in params.items() if k not in ignore}
    if not params:
        return ""

    return "-" + "-".join(f"{k}={v}" for k, v in sorted(params.items()))


def get_params(parser, defaults: str | Path):
    cli_params = parser.parse_args()
    cli_params = Namespace( **{k: v for k, v in vars(cli_params).items() if v is not None} )

    with open(str(defaults), "r") as f:
        params = yaml.safe_load(f)
    default_params = Namespace(**params)
    return Namespace( **( vars(default_params) | vars(cli_params)))


def print_params(params):
    for k, v in sorted(vars(params).items()):
        k = str(k).replace("_", "-")
        print(f"    --{k:<35} {v}")
    print()


def add_defaults(params, defaults_path):
    if defaults_path is not None:
        default_params = load_params(defaults_path)

        for k, v in dict(vars(default_params)).items():
            key_normalized = k.replace("-", "_")

            if not hasattr(params, key_normalized) or (getattr(params, key_normalized) is None):
                setattr(params, key_normalized, v)


def completeness_check(params, allowed_missing=None):
    if allowed_missing is None:
        allowed_missing = []

    missing = []
    for k, v in dict(vars(params)).items():
        if v is not None:
            continue

        if k in allowed_missing:
            continue

        missing += [(k, v)]

    if missing:
        msg = f'{len(missing)} missing values:'
        for k, v in missing:
            msg += "\n\t" + f"{k:<30} : {v}"

        raise ValueError(msg)
