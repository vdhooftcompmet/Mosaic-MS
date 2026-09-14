import argparse
from pathlib import Path


DEFAULT_CONFIG_DIR = Path("../config")
DEFAULT_MN_CONFIG = DEFAULT_CONFIG_DIR / "mn.yaml"
DEFAULT_MS2LDA_CONFIG   = DEFAULT_CONFIG_DIR / "ms2lda.yaml"
DEFAULT_SNAPMS_CONFIG   = DEFAULT_CONFIG_DIR / "snapms.yaml"


def handle_run_mn(args):
    """Executes full Molecular Network (MN) generation workflow."""
    from utils.cli import print_params, add_defaults

    # Fall back to global default path if --defaults is omitted
    args.defaults = args.defaults if args.defaults else DEFAULT_MN_CONFIG
    if args.defaults and Path(args.defaults).exists():
        add_defaults(args, str(args.defaults))

    for key in ["defaults", "func", "command"]:
        if hasattr(args, key):
            delattr(args, key)

    print("> running mn with parameters:\n")
    print_params(args)

    from mn.main import main as run_mn_main
    run_mn_main(args)


def handle_run_ms2lda(args):
    """Executes full MS2LDA workflow."""
    from utils.cli import print_params, add_defaults

    # Fall back to global default path if --defaults is omitted
    args.defaults = args.defaults if args.defaults else DEFAULT_MS2LDA_CONFIG
    if args.defaults and Path(args.defaults).exists():
        add_defaults(args, str(args.defaults))
        
    for key in ["defaults", "func", "command"]:
        if hasattr(args, key):
            delattr(args, key)

    print("> running ms2lda with parameters:\n")
    print_params(args)

    from ms2lda.main import main as run_ms2lda_main
    run_ms2lda_main(args)


def handle_add_ms2lda(args):
    """Adds MS2LDA annotations/model to a target CX graph."""
    print(f"Adding LDA model {args.model} to graph {args.graph}")


def handle_run_snapms(args):
    """Executes full SnapMS compound identification workflow."""
    from utils.cli import print_params, add_defaults

    # Fall back to global default path if --defaults is omitted
    args.defaults = args.defaults if args.defaults else DEFAULT_SNAPMS_CONFIG
    if args.defaults and Path(args.defaults).exists():
        add_defaults(args, str(args.defaults))
        
    for key in ["defaults", "func", "command"]:
        if hasattr(args, key):
            delattr(args, key)

    print("> running snapms with parameters:\n")
    print_params(args)

    from snapms.main import main as run_snapms_main
    run_snapms_main(args)


def handle_add_snapms(args):
    """Appends SnapMS annotations to a target CX graph."""
    print(f"Adding SnapMS results from {args.snapms} to graph {args.graph} (New Copy: {args.new_copy})")


# ==========================================
# Parser Setup
# ==========================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI tool for strata-ms workflows."
    )
    subparsers = parser.add_subparsers(
        dest="command", 
        required=True, 
        help="Sub-command to execute"
    )

    # --------------------------------------
    # 1. run-mn
    # --------------------------------------
    p_build = subparsers.add_parser(
        "run-mn", 
        help="Build a molecular network from MGF spectrum data using SpecReBooted."
    )
    p_build.add_argument(
        "--defaults",
        type=str,
        help="Path to a YAML configuration file containing default parameters.",
        default=None,
    )
    p_build.add_argument(
        "--mgf", "-m",
        type=str,
        help="File path pointing towards the mass spectra (MGF) to be analyzed.",
        default=None,
    )
    p_build.add_argument(
        "--mgf-cleaned",
        type=str,
        help="Save location of cleaned spectra. If it exists, cleaning is skipped.",
        default=None,
    )
    p_build.add_argument(
        "--average-similarity-file",
        type=str,
        help="Path to save or load calculated average similarity matrices (.npz).",
        default=None,
    )
    p_build.add_argument(
        "--similarity-file",
        type=str,
        help="Path to save or load calculated similarity matrices (.npz).",
        default=None,
    )
    p_build.add_argument(
        "--support-file",
        type=str,
        help="Path to save or load calculated SpecReBoot support matrices (.npz).",
        default=None,
    )
    p_build.add_argument(
        "--similarity-type",
        choices=["cos", "modcos", "spec2vec", "ms2deepscore"],
        help="Similarity metric used for networking: cos, modcos, spec2vec, or ms2deepscore.",
        default=None,
    )
    p_build.add_argument(
        "--similarity-threshold",
        type=float,
        help="Minimum similarity threshold required to connect two nodes with an edge.",
        default=None,
    )
    p_build.add_argument(
        "--flash-tolerance",
        type=float,
        help="Mass tolerance error margin (in Da). m/z values within this margin are identical.",
        default=None,
    )
    p_build.add_argument(
        "--binning-decimals",
        type=int,
        help="Precision used to bin spectrum m/z values for SpecReBoot.",
        default=None,
    )
    p_build.add_argument(
        "--max-component-size",
        type=int,
        help="Maximum allowable cluster size in the network.",
        default=None,
    )
    p_build.add_argument(
        "--B",
        type=int,
        help="Number of SpecReBoot bootstrapping iterations per spectrum.",
        default=None,
    )
    p_build.add_argument(
        "--k",
        type=int,
        help="Top-k nearest neighbor spectra considered during bootstrapping support.",
        default=None,
    )
    p_build.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility in bootstrapping calculations.",
        default=None,
    )
    p_build.add_argument(
        "--n-jobs",
        type=int,
        help="Number of parallel execution worker threads.",
        default=None,
    )
    p_build.add_argument(
        "--ms2deepscore-model-path",
        type=str,
        help="Path to trained MS2DeepScore model file.",
        default=None,
    )
    p_build.add_argument(
        "--spec2vec-model-path",
        type=str,
        help="Path to trained Spec2Vec model file.",
        default=None,
    )
    p_build.add_argument(
        "--use-average-similarity",
        action=argparse.BooleanOptionalAction,
        help="Use average similarity from bootstrapping instead of standard pair similarity.",
        default=None,
    )
    p_build.add_argument(
        "--cache-similarity",
        action=argparse.BooleanOptionalAction,
        help="Flag to cache and reuse similarity calculations from a previous run.",
        default=None,
    )
    p_build.add_argument(
        "--base-graph-path",
        type=str,
        help="Save path for generated base network graph (.cx).",
        default=None,
    )
    p_build.add_argument(
        "--threshold-graph-path",
        type=str,
        help="Save path for filtered threshold network graph (.cx).",
        default=None,
    )
    p_build.add_argument(
        "--rescued-graph-path",
        type=str,
        help="Save path for rescued edge network graph (.cx).",
        default=None,
    )
    p_build.set_defaults(func=handle_run_mn)

    # --------------------------------------
    # 2. run-ms2lda
    # --------------------------------------
    p_run_ms2lda = subparsers.add_parser(
        "run-ms2lda", 
        help="Run MS2LDA topic modeling to extract substructural Mass2Motifs from MGF data."
    )
    p_run_ms2lda.add_argument(
        "--defaults",
        type=str,
        help="Path to a YAML configuration file containing default parameters.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--mgf", "-m",
        type=str,
        help="Path to the MGF spectrum data file to analyze.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--mgf-cleaned",
        type=str,
        help="Save path for cleaned spectrum data; skips preprocessing if file exists.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--model-path",
        type=str,
        help="Save path for the output trained LDA model (.bin).",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--motifs-path",
        type=str,
        help="Save path for extracted Mass2Motifs output (.mgf).",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--iterations",
        type=int,
        help="Maximum number of LDA model training iterations to reach convergence.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--nr-of-motifs",
        type=int,
        help="Total number of Mass2Motifs (topics) to extract.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--top-n-words",
        type=int,
        help="Number of top fragment/loss features per topic to include in a motif.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--dataset-acquisition-type",
        choices=["DDA", "DIA"],
        help="Acquisition mode: 'DDA' calculates neutral losses, 'DIA' ignores losses.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--dataset-charge",
        type=int,
        choices=[1, -1],
        help="Ionization polarity charge mode: 1 for positive mode, -1 for negative mode.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--dataset-significant-digits",
        type=int,
        help="Precision for binning peak m/z values into spectral words.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--train-parallel",
        type=int,
        help="Number of parallel execution threads for LDA model training.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--train-workers",
        type=int,
        help="Number of worker processes for LDA model training.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--model-rm_top",
        type=int,
        help="Number of top most frequent global words/peaks to exclude from modeling.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--model-min-cf",
        type=int,
        help="Minimum feature count filter for LDA modeling.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--model-min-df",
        type=int,
        help="Minimum document frequency filter for LDA modeling.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--model-alpha",
        type=float,
        help="Alpha hyperparameter for Dirichlet document-topic distribution.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--model-eta",
        type=float,
        help="Eta hyperparameter for Dirichlet topic-word distribution.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--model-seed",
        type=int,
        help="Random seed for LDA model initialization.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--conv-step-size",
        type=int,
        help="Frequency (in iterations) at which model convergence is evaluated.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--conv-window-size",
        type=int,
        help="Window size used to compute convergence trend metrics.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--conv-threshold",
        type=float,
        help="Target change threshold to signal convergence.",
        default=None,
    )
    p_run_ms2lda.add_argument(
        "--conv-type",
        type=str,
        help="Convergence measurement strategy (e.g., 'perplexity_history').",
        default=None,
    )
    p_run_ms2lda.set_defaults(func=handle_run_ms2lda)

    # --------------------------------------
    # 3. add-ms2lda
    # --------------------------------------
    p_add_ms2lda = subparsers.add_parser(
        "add-ms2lda", 
        help="Integrate extracted MS2LDA motifs into an existing CX graph network."
    )
    p_add_ms2lda.add_argument(
        "--model",
        type=Path,
        help="Path to trained LDA model file.",
        required=True,
    )
    p_add_ms2lda.add_argument(
        "--graph",
        type=Path,
        help="Path to target network CX graph file.",
        required=True,
    )
    p_add_ms2lda.add_argument(
        "--new-copy", "-c",
        action="store_true",
        help="Save result into a new copy of the graph rather than modifying in place.",
        default=False,
    )
    p_add_ms2lda.set_defaults(func=handle_add_ms2lda)

    # --------------------------------------
    # 4. run-snapms
    # --------------------------------------
    p_run_snapms = subparsers.add_parser(
        "run-snapms", 
        help="Annotate molecular networks using SNAP-MS structure database matching."
    )
    p_run_snapms.add_argument(
        "--defaults",
        type=str,
        help="Path to a YAML configuration file containing default parameters.",
        default=None,
    )
    p_run_snapms.add_argument(
        "--graph", "-g",
        type=str,
        help="Path to target molecular network CX graph file to annotate.",
        default=None,
    )
    p_run_snapms.add_argument(
        "--result-folder",
        type=str,
        help="Destination directory to output SNAP-MS annotation results.",
        default=None,
    )
    p_run_snapms.add_argument(
        "--reference-db",
        type=str,
        help="Path to reference structure database (.jsonl with SMILES & neutral_mass).",
        default=None,
    )
    p_run_snapms.add_argument(
        "--ppm-error",
        type=float,
        help="Mass tolerance error window in PPM to query candidate database structures.",
        default=None,
    )
    p_run_snapms.add_argument(
        "--min-cluster-size",
        type=int,
        help="Minimum network cluster size allowed for SNAP-MS annotation.",
        default=None,
    )
    p_run_snapms.add_argument(
        "--max-cluster-size",
        type=int,
        help="Maximum network cluster size allowed for SNAP-MS annotation.",
        default=None,
    )
    p_run_snapms.add_argument(
        "--min-annotation-size",
        type=int,
        help="Minimum size of structural annotation sub-clusters to retain in results.",
        default=None,
    )
    p_run_snapms.add_argument(
        "--cutoff",
        type=float,
        help="Chemical similarity Tanimoto threshold required to connect two structures.",
        default=None,
    )
    p_run_snapms.add_argument(
        "--adduct-list",
        nargs="+",
        help="List of candidate adduct forms to consider during mass matching.",
        default=None,
    )
    p_run_snapms.add_argument(
        "--detect-adduct",
        action=argparse.BooleanOptionalAction,
        help="Automatically detect adduct annotations directly from graph properties.",
        default=None,
    )
    p_run_snapms.set_defaults(func=handle_run_snapms)

    # --------------------------------------
    # 5. add-snapms
    # --------------------------------------
    p_add_snapms = subparsers.add_parser(
        "add-snapms", 
        help="Add external SNAP-MS annotation results into an existing CX graph network."
    )
    p_add_snapms.add_argument(
        "--snapms", "-s",
        type=Path,
        help="Path to SNAP-MS annotation CX file.",
        required=True,
    )
    p_add_snapms.add_argument(
        "--graph", "-g",
        type=Path,
        help="Path to target graph CX file.",
        required=True,
    )
    p_add_snapms.add_argument(
        "--new-copy", "-c",
        action="store_true",
        help="Create a new annotated output copy instead of overwriting original graph.",
        default=False,
    )
    p_add_snapms.set_defaults(func=handle_add_snapms)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()