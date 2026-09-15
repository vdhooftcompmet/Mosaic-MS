import argparse
from pathlib import Path
import time

OPTIMIZED_MODE = True

if OPTIMIZED_MODE:  # super hacky method to save on import time
    import sys
    from unittest.mock import MagicMock

    sys.modules['pynndescent'] = MagicMock()  # saves import time

DEFAULT_CONFIG_DIR = Path("../config")
DEFAULT_MN_CONFIG = DEFAULT_CONFIG_DIR / "mn.yaml"
DEFAULT_MS2LDA_CONFIG = DEFAULT_CONFIG_DIR / "ms2lda.yaml"
DEFAULT_SNAPMS_CONFIG = DEFAULT_CONFIG_DIR / "snapms.yaml"


# ==========================================
# Handlers
# ==========================================

def handle_run_mn(args):
    """Executes full Molecular Network (MN) generation workflow."""
    from utils.cli import print_params, add_defaults

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
    """
    Integrates motif detection and metadata injection into target CX graph.
    Mirrors rule motif_overlap + mn_motif_metadata.py from Snakefile.
    """
    from add_ms2lda.main import main as add_ms2lda_main
    add_ms2lda_main(args)


def handle_run_snapms(args):
    """Executes full SnapMS compound identification workflow."""
    from utils.cli import print_params, add_defaults

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
    """
    Appends SnapMS annotations to a CX graph and flags SMILES.
    Mirrors annotation_metadata scripts in rule snapms from Snakefile.
    """
    from add_snapms.main import main as add_snapms_main

    add_snapms_main(args)


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
        "--defaults", type=str, default=None,
        help="Path to a YAML configuration file containing default parameters."
    )
    p_build.add_argument(
        "--mgf", "-m", type=str, default=None,
        help="File path pointing towards the mass spectra (MGF) to be analyzed."
    )
    p_build.add_argument(
        "--similarity-type", choices=["cos", "modcos", "spec2vec", "ms2deepscore"],
        default=None, help="Similarity metric used for networking."
    )
    p_build.add_argument(
        "--similarity-threshold", type=float, default=None,
        help="Minimum similarity threshold required to connect two nodes with an edge."
    )
    p_build.add_argument(
        "--flash-tolerance", type=float, default=None,
        help="Mass tolerance error margin (in Da)."
    )
    p_build.add_argument(
        "--binning-decimals", type=int, default=None,
        help="Precision used to bin spectrum m/z values."
    )
    p_build.add_argument(
        "--max-component-size", type=int, default=None,
        help="Maximum allowable cluster size in the network."
    )
    p_build.add_argument(
        "--B", type=int, default=None,
        help="Bootstrapping iterations per spectrum."
    )
    p_build.add_argument(
        "--k", type=int, default=None,
        help="Top-k nearest neighbors."
    )
    p_build.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility."
    )
    p_build.add_argument(
        "--n-jobs", type=int, default=None,
        help="Number of parallel worker threads."
    )
    p_build.add_argument(
        "--ms2deepscore-model-path", type=str, default=None,
        help="MS2DeepScore model path."
    )
    p_build.add_argument(
        "--spec2vec-model-path", type=str, default=None,
        help="Spec2Vec model path."
    )
    p_build.add_argument(
        "--use-average-similarity", action=argparse.BooleanOptionalAction,
        default=None, help="Use average similarity from bootstrapping."
    )
    p_build.add_argument(
        "--cache-similarity", action=argparse.BooleanOptionalAction,
        default=None, help="Cache and reuse similarity calculations."
    )
    p_build.add_argument(
        "--base-graph-path", type=str, default=None,
        help="Path for base network graph (.cx)."
    )
    p_build.add_argument(
        "--threshold-graph-path", type=str, default=None,
        help="Path for threshold graph (.cx)."
    )
    p_build.add_argument(
        "--rescued-graph-path", type=str, default=None,
        help="Path for rescued graph (.cx)."
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
        "--defaults", type=str, default=None,
        help="Path to YAML config file."
    )
    p_run_ms2lda.add_argument(
        "--mgf", "-m", type=str, default=None,
        help="Path to MGF spectrum file."
    )
    p_run_ms2lda.add_argument(
        "--model-path", type=str, default=None,
        help="Save path for trained LDA model."
    )
    p_run_ms2lda.add_argument(
        "--motifs-path", type=str, default=None,
        help="Save path for Mass2Motifs (.mgf)."
    )
    p_run_ms2lda.add_argument(
        "--iterations", type=int, default=None,
        help="Max LDA training iterations."
    )
    p_run_ms2lda.add_argument(
        "--nr-of-motifs", type=int, default=None,
        help="Total number of motifs to extract."
    )
    p_run_ms2lda.add_argument(
        "--top-n-words", type=int, default=None,
        help="Number of top fragment/loss features per motif."
    )
    p_run_ms2lda.add_argument(
        "--dataset-acquisition-type", choices=["DDA", "DIA"],
        default=None, help="Acquisition type mode."
    )
    p_run_ms2lda.add_argument(
        "--dataset-charge", type=int, choices=[1, -1],
        default=None, help="Ionization charge polarity."
    )
    p_run_ms2lda.add_argument(
        "--dataset-significant-digits", type=int, default=None,
        help="Precision for binning peak m/z values."
    )
    p_run_ms2lda.add_argument(
        "--train-parallel", type=int, default=None,
        help="Parallel threads for training."
    )
    p_run_ms2lda.add_argument(
        "--train-workers", type=int, default=None,
        help="Worker processes for training."
    )
    p_run_ms2lda.add_argument(
        "--model-rm_top", type=int, default=None,
        help="Number of top global words to exclude."
    )
    p_run_ms2lda.add_argument(
        "--model-min-cf", type=int, default=None,
        help="Minimum feature count filter."
    )
    p_run_ms2lda.add_argument(
        "--model-min-df", type=int, default=None,
        help="Minimum document frequency filter."
    )
    p_run_ms2lda.add_argument(
        "--model-alpha", type=float, default=None,
        help="Alpha hyperparameter for Dirichlet distribution."
    )
    p_run_ms2lda.add_argument(
        "--model-eta", type=float, default=None,
        help="Eta hyperparameter for Dirichlet distribution."
    )
    p_run_ms2lda.add_argument(
        "--model-seed", type=int, default=None,
        help="Random seed for LDA model initialization."
    )
    p_run_ms2lda.add_argument(
        "--conv-step-size", type=int, default=None,
        help="Frequency evaluating convergence."
    )
    p_run_ms2lda.add_argument(
        "--conv-window-size", type=int, default=None,
        help="Window size for convergence trends."
    )
    p_run_ms2lda.add_argument(
        "--conv-threshold", type=float, default=None,
        help="Target change threshold for convergence."
    )
    p_run_ms2lda.add_argument(
        "--conv-type", type=str, default=None,
        help="Convergence measurement strategy."
    )
    p_run_ms2lda.set_defaults(func=handle_run_ms2lda)

    # --------------------------------------
    # 3. add-ms2lda
    # --------------------------------------
    p_add_ms2lda = subparsers.add_parser(
        "add-ms2lda",
        help="Calculate motif overlap and integrate MS2LDA metadata into an existing CX graph network."
    )
    p_add_ms2lda.add_argument(
        "--model", type=Path, required=True,
        help="Path to trained LDA model file (.lda)."
    )
    p_add_ms2lda.add_argument(
        "--graph", "-g", type=Path, required=True,
        help="Path to target network CX graph file."
    )
    p_add_ms2lda.add_argument(
        "--mgf", "-m", type=Path, required=True,
        help="Path to cleaned input MGF file used to compute overlaps."
    )
    p_add_ms2lda.add_argument(
        "--threshold", type=float, default=0.01,
        help="Motif metadata overlap threshold (default: 0.01)."
    )
    p_add_ms2lda.add_argument(
        "--dataset-acquisition-type", type=str, default="DDA",
        help="Aquisition type determines if losses are incorporated. Use the same settings as for run-ms2lda."
    )
    p_add_ms2lda.add_argument(
        "--dataset-significant-digits", type=int, default=2,
        help="Precision for binning peak m/z values."
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
        "--defaults", type=str, default=None,
        help="Path to YAML config file."
    )
    p_run_snapms.add_argument(
        "--graph", "-g", type=str, default=None,
        help="Target molecular network CX graph."
    )
    p_run_snapms.add_argument(
        "--result-folder", type=str, default=None,
        help="Destination directory for output."
    )
    p_run_snapms.add_argument(
        "--reference-db", type=str, default=None,
        help="Reference structure database path."
    )
    p_run_snapms.add_argument(
        "--ppm-error", type=float, default=None,
        help="Mass tolerance error in PPM."
    )
    p_run_snapms.add_argument(
        "--min-cluster-size", type=int, default=None,
        help="Minimum network cluster size."
    )
    p_run_snapms.add_argument(
        "--max-cluster-size", type=int, default=None,
        help="Maximum network cluster size."
    )
    p_run_snapms.add_argument(
        "--min-annotation-size", type=int, default=None,
        help="Minimum annotation cluster size."
    )
    p_run_snapms.add_argument(
        "--cutoff", type=float, default=None,
        help="Tanimoto similarity cutoff threshold."
    )
    p_run_snapms.add_argument(
        "--adduct-list", nargs="+", default=None,
        help="List of candidate adduct forms."
    )
    p_run_snapms.add_argument(
        "--detect-adduct", action=argparse.BooleanOptionalAction, default=None,
        help="Automatically detect adduct annotations directly from graph properties."
    )
    p_run_snapms.set_defaults(func=handle_run_snapms)

    # --------------------------------------
    # 5. add-snapms
    # --------------------------------------
    p_add_snapms = subparsers.add_parser(
        "add-snapms",
        help="Relay annotation metadata and flag SMILES on target CX graph."
    )
    p_add_snapms.add_argument(
        "--graph", "-g", type=Path, required=True,
        help="Path to target network CX graph file."
    )
    p_add_snapms.add_argument(
        "--snapms", "-a", type=Path, required=True,
        help="Path to SNAP-MS annotation results directory containing output files."
    )
    p_add_snapms.set_defaults(func=handle_add_snapms)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()