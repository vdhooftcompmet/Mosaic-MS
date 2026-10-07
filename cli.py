import argparse
from pathlib import Path

import yaml

from src.utils.configs import (
    AddMAGConfig,
    AddMS2LDAConfig,
    AddSNAPMSConfig,
    MNConfig,
    MS2LDAConfig,
    SNAPMSConfig,
)

REPO = Path(__file__).parent.resolve()
DEFAULT_MN_CONFIG = REPO / "config" / "mn.yaml"
DEFAULT_MS2LDA_CONFIG = REPO / "config" / "ms2lda.yaml"
DEFAULT_ADD_MS2LDA_CONFIG = REPO / "config" / "add_ms2lda.yaml"
DEFAULT_ADD_MAG_CONFIG = REPO / "config" / "add_mag.yaml"
DEFAULT_SNAPMS_CONFIG = REPO / "config" / "snapms.yaml"
DEFAULT_ADD_SNAPMS_CONFIG = REPO / "config" / "add_snapms.yaml"


def prepare_args(
    args: argparse.Namespace, default_config: Path
) -> argparse.Namespace:
    """Applies YAML defaults, strips control attributes, and logs parameters."""

    args.defaults = args.defaults if getattr(args, "defaults", None) else str(default_config)
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


def handle_run_mn(args) -> None:
    """Executes full Molecular Network (MN) generation workflow."""
    args = prepare_args(args, DEFAULT_MN_CONFIG)
    mn_config = MNConfig(**dict(vars(args)))
    mn_config.display()

    from src.mn.main import main as run_mn_main

    run_mn_main(mn_config)


def handle_run_ms2lda(args) -> None:
    """Executes full MS2LDA workflow."""
    args = prepare_args(args, DEFAULT_MS2LDA_CONFIG)
    ms2lda_config = MS2LDAConfig(**dict(vars(args)))
    ms2lda_config.display()

    from src.ms2lda.main import main as run_ms2lda_main

    run_ms2lda_main(ms2lda_config)


def handle_add_ms2lda(args) -> None:
    """
    Integrates motif detection and metadata injection into target CX graph.
    Mirrors rule motif_overlap + mn_motif_metadata.py from Snakefile.
    """
    args = prepare_args(args, DEFAULT_ADD_MS2LDA_CONFIG)
    add_ms2lda_config = AddMS2LDAConfig(**dict(vars(args)))
    add_ms2lda_config.display()

    from src.add_ms2lda.main import main as add_ms2lda_main

    add_ms2lda_main(add_ms2lda_config)


def handle_add_mag(args) -> None:
    """Annotates Mass2Motifs using Motif Annotation Guidance (MAG)."""
    args = prepare_args(args, DEFAULT_ADD_MAG_CONFIG)
    add_mag_config = AddMAGConfig(**dict(vars(args)))
    add_mag_config.display()

    from src.ms2lda.mag import main as add_mag_main

    add_mag_main(add_mag_config)


def handle_run_snapms(args) -> None:
    """Executes full SnapMS compound identification workflow."""
    args = prepare_args(args, DEFAULT_SNAPMS_CONFIG)
    snapms_config = SNAPMSConfig(**dict(vars(args)))
    snapms_config.display()

    from src.snapms.main import main as run_snapms_main

    run_snapms_main(snapms_config)


def handle_add_snapms(args) -> None:
    """
    Appends SnapMS annotations to a CX graph and flags SMILES.
    Mirrors annotation_metadata scripts in rule snapms from Snakefile.
    """
    args = prepare_args(args, DEFAULT_ADD_SNAPMS_CONFIG)
    add_snapms_config = AddSNAPMSConfig(**dict(vars(args)))
    add_snapms_config.display()

    from src.add_snapms.main import main as add_snapms_main

    add_snapms_main(add_snapms_config)


def handle_run_all(args) -> None:
    """Executes the complete Mosaic-MS pipeline sequentially."""

    # 1. Run Molecular Networking
    print("\n--- [Step 1/6] Running Molecular Networking ---")
    handle_run_mn(
        argparse.Namespace(
            mgf=args.mgf,
            defaults=args.mn_defaults,
            base_graph_path=str(args.base_graph_path),
        )
    )

    # 2. Run MS2LDA
    print("\n--- [Step 2/6] Running MS2LDA ---")
    handle_run_ms2lda(
        argparse.Namespace(
            mgf=args.mgf,
            defaults=args.ms2lda_defaults,
            model_path=str(args.ms2lda_model_path),
            motifs_path=str(args.ms2lda_motifs_path),
        )
    )

    # 3. Add MS2LDA Results to Graph
    print("\n--- [Step 3/6] Adding MS2LDA Results to Graph ---")
    handle_add_ms2lda(
        argparse.Namespace(
            defaults=args.add_ms2lda_defaults,
            graph=args.base_graph_path,
            model=args.ms2lda_model_path,
            threshold=args.ms2lda_threshold,
        )
    )

    # 4. Add MAG (Motif Annotation Guidance)
    print("\n--- [Step 4/6] Running Motif Annotation Guidance (MAG) ---")
    # FIXME: currently not allowed due to missing library in standard files
    handle_add_mag(
        argparse.Namespace(
            defaults=args.add_mag_defaults,
            motifs=args.ms2lda_motifs_path,
        )
    )

    # 5. Run SNAP-MS
    print("\n--- [Step 5/6] Running SNAP-MS ---")
    handle_run_snapms(
        argparse.Namespace(
            graph=str(args.base_graph_path),
            defaults=args.snapms_defaults,
            result_folder=str(args.snapms_result_folder),
        )
    )

    # 6. Add SNAP-MS Results to Graph
    print("\n--- [Step 6/6] Adding SNAP-MS Results to Graph ---")
    handle_add_snapms(
        argparse.Namespace(
            defaults=args.add_snapms_defaults,
            graph=args.base_graph_path,
            snapms=args.snapms_result_folder,
        )
    )

    print("\n Pipeline execution completed successfully!")


# ==========================================
# Parser Setup
# ==========================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI tool for strata-ms workflows.")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-command to execute"
    )

    # --------------------------------------
    # 1. run-mn
    # --------------------------------------
    p_build = subparsers.add_parser(
        "run-mn",
        help="Build a molecular network from MGF spectrum data using SpecReBooted.",
    )
    p_build.add_argument(
        "--defaults",
        type=str,
        default=None,
        help="Path to a YAML configuration file containing default parameters.",
    )
    p_build.add_argument(
        "--mgf",
        "-m",
        type=str,
        default=None,
        help="File path pointing towards the mass spectra (MGF) to be analyzed.",
    )
    p_build.add_argument(
        "--similarity-type",
        choices=["cos", "modcos", "spec2vec", "ms2deepscore"],
        default=None,
        help="Similarity metric used for networking.",
    )
    p_build.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Minimum similarity threshold required to connect two nodes with an edge.",
    )
    p_build.add_argument(
        "--flash-tolerance",
        type=float,
        default=None,
        help="Mass tolerance error margin (in Da).",
    )
    p_build.add_argument(
        "--binning-decimals",
        type=int,
        default=None,
        help="Precision used to bin spectrum m/z values.",
    )
    p_build.add_argument(
        "--max-component-size",
        type=int,
        default=None,
        help="Maximum allowable cluster size in the network.",
    )
    p_build.add_argument(
        "--B", type=int, default=None, help="Bootstrapping iterations per spectrum."
    )
    p_build.add_argument("--k", type=int, default=None, help="Top-k nearest neighbors.")
    p_build.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    p_build.add_argument(
        "--ms2deepscore-model-path",
        type=str,
        default=None,
        help="MS2DeepScore model path.",
    )
    p_build.add_argument(
        "--spec2vec-model-path", type=str, default=None, help="Spec2Vec model path."
    )
    p_build.add_argument(
        "--use-average-similarity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use average similarity from bootstrapping.",
    )
    p_build.add_argument(
        "--cache-similarity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache and reuse similarity calculations.",
    )
    p_build.add_argument(
        "--base-graph-path",
        type=str,
        default=None,
        help="Path for base network graph (.cx).",
    )
    p_build.add_argument(
        "--threshold-graph-path",
        type=str,
        default=None,
        help="Path for threshold graph (.cx).",
    )
    p_build.add_argument(
        "--rescued-graph-path",
        type=str,
        default=None,
        help="Path for rescued graph (.cx).",
    )
    p_build.set_defaults(func=handle_run_mn)

    # --------------------------------------
    # 2. run-ms2lda
    # --------------------------------------
    p_run_ms2lda = subparsers.add_parser(
        "run-ms2lda",
        help="Run MS2LDA topic modeling to extract substructural Mass2Motifs from MGF data.",
    )
    p_run_ms2lda.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    p_run_ms2lda.add_argument(
        "--mgf", "-m", type=str, default=None, help="Path to MGF spectrum file."
    )
    p_run_ms2lda.add_argument(
        "--model-path", type=str, default=None, help="Save path for trained LDA model."
    )
    p_run_ms2lda.add_argument(
        "--motifs-path",
        type=str,
        default=None,
        help="Save path for Mass2Motifs (.mgf).",
    )
    p_run_ms2lda.add_argument(
        "--iterations", type=int, default=None, help="Max LDA training iterations."
    )
    p_run_ms2lda.add_argument(
        "--nr-of-motifs",
        type=int,
        default=None,
        help="Total number of motifs to extract.",
    )
    p_run_ms2lda.add_argument(
        "--top-n-words",
        type=int,
        default=None,
        help="Number of top fragment/loss features per motif.",
    )
    p_run_ms2lda.add_argument(
        "--dataset-acquisition-type",
        choices=["DDA", "DIA"],
        default=None,
        help="Acquisition type mode.",
    )
    p_run_ms2lda.add_argument(
        "--dataset-charge",
        type=int,
        choices=[1, -1],
        default=None,
        help="Ionization charge polarity.",
    )
    p_run_ms2lda.add_argument(
        "--dataset-significant-digits",
        type=int,
        default=None,
        help="Precision for binning peak m/z values.",
    )
    p_run_ms2lda.add_argument(
        "--train-parallel",
        type=int,
        default=None,
        help="Parallel threads for training.",
    )
    p_run_ms2lda.add_argument(
        "--train-workers", type=int, default=None, help="Worker processes for training."
    )
    p_run_ms2lda.add_argument(
        "--model-rm_top",
        type=int,
        default=None,
        help="Number of top global words to exclude.",
    )
    p_run_ms2lda.add_argument(
        "--model-min-cf", type=int, default=None, help="Minimum feature count filter."
    )
    p_run_ms2lda.add_argument(
        "--model-min-df",
        type=int,
        default=None,
        help="Minimum document frequency filter.",
    )
    p_run_ms2lda.add_argument(
        "--model-alpha",
        type=float,
        default=None,
        help="Alpha hyperparameter for Dirichlet distribution.",
    )
    p_run_ms2lda.add_argument(
        "--model-eta",
        type=float,
        default=None,
        help="Eta hyperparameter for Dirichlet distribution.",
    )
    p_run_ms2lda.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Random seed for LDA model initialization.",
    )
    p_run_ms2lda.add_argument(
        "--conv-step-size",
        type=int,
        default=None,
        help="Frequency evaluating convergence.",
    )
    p_run_ms2lda.add_argument(
        "--conv-window-size",
        type=int,
        default=None,
        help="Window size for convergence trends.",
    )
    p_run_ms2lda.add_argument(
        "--conv-threshold",
        type=float,
        default=None,
        help="Target change threshold for convergence.",
    )
    p_run_ms2lda.add_argument(
        "--conv-type", type=str, default=None, help="Convergence measurement strategy."
    )
    p_run_ms2lda.set_defaults(func=handle_run_ms2lda)

    # --------------------------------------
    # 3. add-ms2lda
    # --------------------------------------
    p_add_ms2lda = subparsers.add_parser(
        "add-ms2lda",
        help="Calculate motif overlap and integrate MS2LDA metadata into an existing CX graph network.",
    )
    p_add_ms2lda.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    p_add_ms2lda.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to trained LDA model file (.lda).",
    )
    p_add_ms2lda.add_argument(
        "--graph",
        "-g",
        required=True,
        type=str,
        help="Path to target network CX graph file.",
    )
    p_add_ms2lda.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Motif metadata overlap threshold.",
    )
    p_add_ms2lda.set_defaults(func=handle_add_ms2lda)

    # --------------------------------------
    # 4. add-mag
    # --------------------------------------
    p_add_mag = subparsers.add_parser(
        "add-mag",
        help="Annotate extracted Mass2Motifs using Motif Annotation Guidance (MAG).",
    )
    p_add_mag.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    p_add_mag.add_argument(
        "--motifs",
        required=True,
        type=str,
        help="Path to extracted Mass2Motifs MGF file.",
    )
    p_add_mag.add_argument(
        "--spec2vec-model-path",
        type=str,
        default=None,
        help="Path to Spec2Vec model file.",
    )
    p_add_mag.add_argument(
        "--library",
        type=str,
        default=None,
        help="Path to reference library MGF file.",
    )
    p_add_mag.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity score threshold.",
    )
    p_add_mag.add_argument(
        "--cluster-delta",
        type=float,
        default=None,
        help="Cluster delta threshold.",
    )
    p_add_mag.add_argument(
        "--criterium",
        type=str,
        default=None,
        help="Selection criterium metric.",
    )
    p_add_mag.set_defaults(func=handle_add_mag)

    # --------------------------------------
    # 5. run-snapms
    # --------------------------------------
    p_run_snapms = subparsers.add_parser(
        "run-snapms",
        help="Annotate molecular networks using SNAP-MS structure database matching.",
    )
    p_run_snapms.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    p_run_snapms.add_argument(
        "--graph",
        "-g",
        type=str,
        default=None,
        help="Target molecular network CX graph.",
    )
    p_run_snapms.add_argument(
        "--result-folder",
        type=str,
        default=None,
        help="Destination directory for output.",
    )
    p_run_snapms.add_argument(
        "--reference-db",
        type=str,
        default=None,
        help="Reference structure database path.",
    )
    p_run_snapms.add_argument(
        "--ppm-error", type=float, default=None, help="Mass tolerance error in PPM."
    )
    p_run_snapms.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Minimum network cluster size.",
    )
    p_run_snapms.add_argument(
        "--max-cluster-size",
        type=int,
        default=None,
        help="Maximum network cluster size.",
    )
    p_run_snapms.add_argument(
        "--min-annotation-size",
        type=int,
        default=None,
        help="Minimum annotation cluster size.",
    )
    p_run_snapms.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Tanimoto similarity cutoff threshold.",
    )
    p_run_snapms.add_argument(
        "--adduct-list", nargs="+", default=None, help="List of candidate adduct forms."
    )
    p_run_snapms.add_argument(
        "--detect-adduct",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatically detect adduct annotations directly from graph properties.",
    )
    p_run_snapms.set_defaults(func=handle_run_snapms)

    # --------------------------------------
    # 6. add-snapms
    # --------------------------------------
    p_add_snapms = subparsers.add_parser(
        "add-snapms",
        help="Relay annotation metadata and flag SMILES on target CX graph.",
    )
    p_add_snapms.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    p_add_snapms.add_argument(
        "--graph",
        "-g",
        required=True,
        type=str,
        help="Path to target network CX graph file.",
    )
    p_add_snapms.add_argument(
        "--snapms",
        "-a",
        required=True,
        type=str,
        help="Path to SNAP-MS annotation results directory containing output files.",
    )
    p_add_snapms.set_defaults(func=handle_add_snapms)

    # --------------------------------------
    # 7. run-all
    # --------------------------------------
    p_run_all = subparsers.add_parser(
        "run-all", help="Execute the full end-to-end Mosaic-MS pipeline sequentially."
    )
    p_run_all.add_argument(
        "--mgf", "-m", type=str, required=True, help="Path to input MGF spectrum file."
    )
    p_run_all.add_argument(
        "--base-graph-path",
        type=str,
        default="results/base.cx",
        help="Output path for intermediate and final network CX graph file (default: results/base.cx).",
    )
    p_run_all.add_argument(
        "--ms2lda-model-path",
        type=str,
        default="results/model.lda",
        help="Output path for trained MS2LDA model file (default: results/model.lda).",
    )
    p_run_all.add_argument(
        "--ms2lda-motifs-path",
        type=str,
        default="results/motifs.mgf",
        help="Output path for extracted MS2LDA Mass2Motifs (default: results/motifs.mgf).",
    )
    p_run_all.add_argument(
        "--snapms-result-folder",
        type=str,
        default="results/snapms",
        help="Output directory for SNAP-MS intermediate results (default: results/snapms).",
    )
    p_run_all.add_argument(
        "--mn-defaults",
        type=str,
        default=None,
        help="Path to YAML config file for MN step.",
    )
    p_run_all.add_argument(
        "--ms2lda-defaults",
        type=str,
        default=None,
        help="Path to YAML config file for MS2LDA step.",
    )
    p_run_all.add_argument(
        "--add-ms2lda-defaults",
        type=str,
        default=None,
        help="Path to YAML config file for Add MS2LDA step.",
    )
    p_run_all.add_argument(
        "--add-mag-defaults",
        type=str,
        default=None,
        help="Path to YAML config file for Add MAG step.",
    )
    p_run_all.add_argument(
        "--snapms-defaults",
        type=str,
        default=None,
        help="Path to YAML config file for SNAP-MS step.",
    )
    p_run_all.add_argument(
        "--add-snapms-defaults",
        type=str,
        default=None,
        help="Path to YAML config file for Add SNAP-MS step.",
    )
    p_run_all.add_argument(
        "--ms2lda-threshold",
        type=float,
        default=0.01,
        help="MS2LDA motif overlap threshold (default: 0.01).",
    )
    p_run_all.set_defaults(func=handle_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
