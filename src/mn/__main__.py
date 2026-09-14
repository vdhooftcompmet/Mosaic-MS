import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from utils.cli import completeness_check, add_defaults, print_params


def cli() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("--defaults", type=str)
    
    parser.add_argument("--folder", type=str)
    parser.add_argument("--mgf", type=str)

    parser.add_argument("--average-similarity-file", type=str)
    parser.add_argument("--similarity-file", type=str)
    parser.add_argument("--support-file", type=str)

    parser.add_argument("--similarity-type", choices=["cos", "modcos", "spec2vec", "ms2deepscore"])
    parser.add_argument("--similarity-threshold", type=float)
    parser.add_argument("--flash-tolerance", type=float)
    parser.add_argument("--binning-decimals", type=int)
    parser.add_argument("--max-component-size", type=int)

    parser.add_argument("--B", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-jobs", type=int)

    parser.add_argument("--ms2deepscore-model-path", type=str)
    parser.add_argument("--spec2vec-model-path", type=str)

    parser.add_argument("--use-average-similarity", action=BooleanOptionalAction)
    parser.add_argument("--cache-similarity", action=BooleanOptionalAction)

    parser.add_argument("--base-graph-path", type=str)
    parser.add_argument("--threshold-graph-path", type=str)
    parser.add_argument("--rescued-graph-path", type=str)

    return parser
    

def derive_file_names(params) -> None:
    missing = (lambda p: not hasattr(params, p) or not getattr(params, p))

    if missing("folder"):
        params.folder = "."

    if missing("mgf_cleaned"):
        params.mgf_cleaned = f"{Path(params.mgf).stem}_mn_cleaned.mgf"

    if missing("support_file"):
        params.support_file = f"{params.similarity_type}_{Path(params.mgf_cleaned).stem}_support.npz"

    if missing("similarity_file"):
        params.similarity_file = f"{params.similarity_type}_{Path(params.mgf_cleaned).stem}_similarity.npz"

    if missing("average_similarity_file"):
        params.average_similarity_file = f"{params.similarity_type}_{Path(params.mgf_cleaned).stem}_average_similarity.npz"

    if missing("base_graph_path"):
        params.base_graph_path = f"{params.similarity_type}_{Path(params.mgf_cleaned).stem}_base.cx"

    if missing("threshold_graph_path"):
        params.threshold_graph_path = f"{params.similarity_type}_{Path(params.mgf_cleaned).stem}_threshold.cx"

    if missing("rescued_graph_path"):
        params.rescued_graph_path = f"{params.similarity_type}_{Path(params.mgf_cleaned).stem}_rescued.cx"

    if not Path(params.mgf_cleaned).is_absolute():
        params.mgf_cleaned = str( Path(params.folder) / params.mgf_cleaned)

    if not Path(params.support_file).is_absolute():
        params.support_file = str( Path(params.folder) / params.support_file)

    if not Path(params.similarity_file).is_absolute():
        params.similarity_file = str( Path(params.folder) / params.similarity_file)

    if not Path(params.average_similarity_file).is_absolute():
        params.average_similarity_file = str( Path(params.folder) / params.average_similarity_file)

    if not Path(params.base_graph_path).is_absolute():
        params.base_graph_path = str( Path(params.folder) / params.base_graph_path)

    if not Path(params.threshold_graph_path).is_absolute():
        params.threshold_graph_path = str( Path(params.folder) / params.threshold_graph_path)

    if not Path(params.rescued_graph_path).is_absolute():
        params.rescued_graph_path = str( Path(params.folder) / params.rescued_graph_path)


if __name__ == "__main__":

    parser = cli()
    params = parser.parse_args()
    add_defaults(params, params.defaults)
    delattr(params, "defaults")
    derive_file_names(params)
    completeness_check(params, allowed_missing=["spec2vec_model_path", "ms2deepscore_model_path", "mgf"])

    if params.similarity_type == "spec2vec":
        if not hasattr(params, "spec2vec_model_path") or not getattr(params, "spec2vec_model_path"):
            raise ValueError('parameter "spec2vec_model_path" required when networking with spec2vec')
        
    if params.similarity_type == "ms2deepscore":
        if not hasattr(params, "ms2deepscore_model_path") or not getattr(params, "ms2deepscore_model_path"):
            raise ValueError('parameter "ms2deepscore_model_path" required when networking with ms2deepscore')
        
    print("> running mn with parameters:\n")
    print_params(params)

    from mn.main import main
    main(params)
