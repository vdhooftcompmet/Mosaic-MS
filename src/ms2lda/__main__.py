import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from argparse import ArgumentParser
from pathlib import Path

from utils.cli import print_params , add_defaults, completeness_check


def cli() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("--defaults", type=str)

    parser.add_argument("--folder", type=str)
    parser.add_argument("--mgf", type=str)
    parser.add_argument("--mgf-cleaned", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--motifs-path", type=str)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--nr-of-motifs", type=int)
    parser.add_argument("--top-n-words", type=int)

    parser.add_argument("--dataset-acquisition-type", type=str)
    parser.add_argument("--dataset-charge", type=int)
    parser.add_argument("--dataset-significant-digits", type=int)

    parser.add_argument("--train-parallel", type=int)
    parser.add_argument("--train-workers", type=int)

    parser.add_argument("--model-rm_top", type=int)
    parser.add_argument("--model-min-cf", type=int)
    parser.add_argument("--model-min-df", type=int)
    parser.add_argument("--model-alpha", type=float)
    parser.add_argument("--model-eta", type=float)
    parser.add_argument("--model-seed", type=int)

    parser.add_argument("--conv-step-size", type=int)
    parser.add_argument("--conv-window-size", type=int)
    parser.add_argument("--conv-threshold", type=float)
    parser.add_argument("--conv-type", type=str)
    return parser


def derive_file_names(params) -> None:
    missing = (lambda p: not hasattr(params, p) or not getattr(params, p))

    if missing("folder"):
        params.folder = "."

    if missing("mgf_cleaned"):
        params.mgf_cleaned = f"{Path(params.mgf).stem}_ms2lda_cleaned.mgf"

    if missing("model_path"):
        params.model_path = f"{Path(params.mgf_cleaned).stem}_model.bin"

    if missing("motifs_path"):
        params.motifs_path = f"{Path(params.mgf_cleaned).stem}_motifs.mgf"
    
    if not Path(params.model_path).is_absolute():
        params.model_path = str( Path(params.folder) / params.model_path)

    if not Path(params.motifs_path).is_absolute():
        params.motifs_path = str( Path(params.folder) / params.motifs_path)

    if not Path(params.mgf_cleaned).is_absolute():
        params.mgf_cleaned = str( Path(params.folder) / params.mgf_cleaned)


if __name__ == "__main__":

    parser = cli()
    params = parser.parse_args()
    add_defaults(params, params.defaults)
    delattr(params, "defaults")
    derive_file_names(params)
    completeness_check(params)
    print("> running ms2lda with parameters:\n")
    print_params(params)

    from ms2lda.main import main
    main(params)
