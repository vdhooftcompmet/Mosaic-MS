import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils.cli import print_params
from utils.cli import print_params, add_defaults, completeness_check

from argparse import  ArgumentParser, BooleanOptionalAction


def cli():
    parser = ArgumentParser()
    parser.add_argument("--defaults", type=str)

    parser.add_argument("--graph", type=str)
    parser.add_argument("--result-folder", type=str)
    parser.add_argument("--reference-db", type=str)

    parser.add_argument("--ppm-error", type=float)
    parser.add_argument("--min-cluster-size", type=int)
    parser.add_argument("--max-cluster-size", type=int)
    parser.add_argument("--min-annotation-size", type=int)
    parser.add_argument("--cutoff", type=float)

    parser.add_argument("--adduct-list", nargs="+")
    parser.add_argument("--detect-adduct", action=BooleanOptionalAction)

    return parser


def derive_file_names(params) -> None:
    pass


if __name__ == "__main__":
    
    parser = cli()
    params = parser.parse_args()
    add_defaults(params, params.defaults)
    delattr(params, "defaults")
    derive_file_names(params)
    completeness_check(params)
    print("> running snapms with parameters:\n")
    print_params(params)

    from snapms.main import main
    main(params)
