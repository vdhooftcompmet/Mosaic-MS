from argparse import Namespace, ArgumentParser, BooleanOptionalAction
from pathlib import Path
import yaml
from utils.folders import load_params

def ms2lda_parser() -> ArgumentParser:
    parser = ArgumentParser(description="MS2LDA pipeline configuration")

    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--nr_of_motifs", type=int)
    parser.add_argument("--top_n_words", type=int)
    parser.add_argument("--out", type=Path, required=True)

    parser.add_argument("--dataset_acquisition_type", type=str)
    parser.add_argument("--dataset_charge", type=int)
    parser.add_argument("--dataset_significant_digits", type=int)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_output_folder", type=Path)

    parser.add_argument("--train_parallel", type=int)
    parser.add_argument("--train_workers", type=int)

    parser.add_argument("--model_rm_top", type=int)
    parser.add_argument("--model_min_cf", type=int)
    parser.add_argument("--model_min_df", type=int)
    parser.add_argument("--model_alpha", type=float)
    parser.add_argument("--model_eta", type=float)
    parser.add_argument("--model_seed", type=int)

    parser.add_argument("--conv_step_size", type=int)
    parser.add_argument("--conv_window_size", type=int)
    parser.add_argument("--conv_threshold", type=float)
    parser.add_argument("--conv_type", type=str)

    parser.add_argument("--ann_criterium", type=str)
    parser.add_argument("--ann_cosine_similarity", type=float)
    parser.add_argument("--ann_n_mols_retrieved", type=int)
    parser.add_argument("--ann_s2v_model_path", type=Path)
    parser.add_argument("--ann_s2v_library_embeddings", type=Path)
    parser.add_argument("--ann_s2v_library_db", type=Path)

    parser.add_argument("--prep_min_mz", type=float)
    parser.add_argument("--prep_max_mz", type=float)
    parser.add_argument("--prep_max_frags", type=int)
    parser.add_argument("--prep_min_frags", type=int)
    parser.add_argument("--prep_min_intensity", type=float)
    parser.add_argument("--prep_max_intensity", type=float)

    parser.add_argument("--fp_type", type=str)
    parser.add_argument("--fp_threshold", type=float)

    return parser


def mn_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="MN pipeline configuration"
    )
    parser.add_argument("--networking-method", choices=["cos", "modcos", "s2v", "ms2ds"])
    parser.add_argument("--run-specreboot", action=BooleanOptionalAction)

    parser.add_argument("--data", type=Path)
    parser.add_argument("--ms2dp-model", type=Path)
    parser.add_argument("--spec2vec-model", type=Path)

    parser.add_argument("--out", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--cleaned-mgf")

    parser.add_argument("--decimals", type=int)
    parser.add_argument("--B", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--label-mode", choices=["feature", "scan", "internal"])

    parser.add_argument("--sim-threshold", type=float)
    parser.add_argument("--sim-threshold-ms2dp", type=float)

    parser.add_argument("--flash-tolerance", type=float)

    parser.add_argument("--support-threshold", type=float)
    parser.add_argument("--max-component-size", type=int)
    parser.add_argument("--support-core", type=float)
    parser.add_argument("--sim-rescue-min", type=float)
    parser.add_argument("--support-rescue", type=float)

    return parser


def snapms_parser():
    parser = ArgumentParser(
        description="SNAP-MS pipeline configuration"
    )

    # File paths
    parser.add_argument("--data", type=Path)
    parser.add_argument("--reference-db", type=Path)
    parser.add_argument("--output-path", type=Path)

    # Numeric parameters
    parser.add_argument("--ppm-error", type=float)
    parser.add_argument("--min-gnps-cluster-size", type=int)
    parser.add_argument("--max-gnps-cluster-size", type=int)
    parser.add_argument("--min-atlas-annotation-cluster-size", type=int)
    parser.add_argument("--min-compound-group-count", type=int)
    parser.add_argument("--max-node-count", type=int)
    parser.add_argument("--max-edge-count", type=int)
    parser.add_argument("--cutoff", type=float)

    # Lists / flags
    parser.add_argument("--adduct-list", nargs="+")
    parser.add_argument("--remove-duplicates", action=BooleanOptionalAction)
    parser.add_argument("--compress-output", action=BooleanOptionalAction)

    # Job / workflow
    parser.add_argument("--job-id")

    # Atlas filter
    parser.add_argument("--atlas-filter", choices=["full",  "bacteria",  "fungi",  "custom",  "coconut"])
    parser.add_argument("--custom-filter")
    parser.add_argument("--hidden-fraction")

    return parser


def mn_metadata_parser():
    parser = ArgumentParser(
        description="adds metadata to existing graphs"
    )

    # File paths
    parser.add_argument("--data", type=Path)
    parser.add_argument("--reference-db", type=Path)
    parser.add_argument("--output-path", type=Path)

    # Numeric parameters
    parser.add_argument("--ppm-error", type=float)
    parser.add_argument("--min-gnps-cluster-size", type=int)
    parser.add_argument("--max-gnps-cluster-size", type=int)
    parser.add_argument("--min-atlas-annotation-cluster-size", type=int)
    parser.add_argument("--min-compound-group-count", type=int)
    parser.add_argument("--max-node-count", type=int)
    parser.add_argument("--max-edge-count", type=int)
    parser.add_argument("--cutoff", type=float)

    # Lists / flags
    parser.add_argument("--adduct-list", nargs="+")
    parser.add_argument("--remove-duplicates", action=BooleanOptionalAction)
    parser.add_argument("--compress-output", action=BooleanOptionalAction)

    # Job / workflow
    parser.add_argument("--job-id")

    # Atlas filter
    parser.add_argument("--atlas-filter", choices=["full",  "bacteria",  "fungi",  "custom",  "coconut"])
    parser.add_argument("--custom-filter")
    parser.add_argument("--hidden-fraction")

    return parser


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

