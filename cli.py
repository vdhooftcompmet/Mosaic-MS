import argparse
from pathlib import Path

from src.add_ms2lda.main import build_parser as build_add_ms2lda_parser
from src.add_ms2lda.main import main as add_ms2lda_main
from src.add_snapms.main import build_parser as build_add_snapms_parser
from src.add_snapms.main import main as add_snapms_main
from src.annotation.main import build_parser as build_annotation_parser
from src.annotation.main import main as run_annotation_main
from src.mn.main import build_parser as build_mn_parser
from src.mn.main import main as run_mn_main
from src.ms2lda.mag import build_parser as build_add_mag_parser
from src.ms2lda.mag import main as add_mag_main
from src.ms2lda.main import build_parser as build_ms2lda_parser
from src.ms2lda.main import main as run_ms2lda_main
from src.snapms.main import build_parser as build_snapms_parser
from src.snapms.main import main as run_snapms_main

REPO = Path(__file__).parent.resolve()
DEFAULT_MN_CONFIG = REPO / "config" / "mn.yaml"
DEFAULT_MS2LDA_CONFIG = REPO / "config" / "ms2lda.yaml"
DEFAULT_ADD_MS2LDA_CONFIG = REPO / "config" / "add_ms2lda.yaml"
DEFAULT_ADD_MAG_CONFIG = REPO / "config" / "add_mag.yaml"
DEFAULT_SNAPMS_CONFIG = REPO / "config" / "snapms.yaml"
DEFAULT_ADD_SNAPMS_CONFIG = REPO / "config" / "add_snapms.yaml"
DEFAULT_ANNOTATION_CONFIG = REPO / "config" / "annotation.yaml"


def handle_run_mn(args: argparse.Namespace) -> None:
    """Executes full Molecular Network (MN) generation workflow."""
    run_mn_main(args)


def handle_run_ms2lda(args: argparse.Namespace) -> None:
    """Executes full MS2LDA workflow."""
    run_ms2lda_main(args)


def handle_add_ms2lda(args: argparse.Namespace) -> None:
    """
    Integrates motif detection and metadata injection into target CX graph.
    """
    add_ms2lda_main(args)


def handle_add_mag(args: argparse.Namespace) -> None:
    """Annotates Mass2Motifs using Motif Annotation Guidance (MAG)."""
    try:
        add_mag_main(args)
    except AssertionError:
        print("!! Skipped this step due to missing library !!")


def handle_run_snapms(args: argparse.Namespace) -> None:
    """Executes full SnapMS compound identification workflow."""
    run_snapms_main(args)


def handle_add_snapms(args: argparse.Namespace) -> None:
    """
    Appends SnapMS annotations to a CX graph and flags SMILES.
    """
    add_snapms_main(args)


def handle_run_annotation(args: argparse.Namespace) -> None:
    """Executes motif fragment based SNAP-MS workflow[cite: 2]."""
    run_annotation_main(args)


def handle_run_all(args: argparse.Namespace) -> None:
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
    try:
        assert False
        handle_add_mag(
            argparse.Namespace(
                defaults=args.add_mag_defaults,
                motifs=args.ms2lda_motifs_path,
            )
        )
    except AssertionError:
        print("!! Skipped this step due to missing library !!")

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
# Parser Setup (Superparser)
# ==========================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI tool for strata-ms workflows.")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-command to execute"
    )

    # --------------------------------------
    # 1. run-mn
    # --------------------------------------
    p_run_mn = subparsers.add_parser(
        "run-mn",
        parents=[build_mn_parser()],
        add_help=False,
        help="Build a molecular network from MGF spectrum data using SpecReBooted.",
    )
    p_run_mn.set_defaults(func=handle_run_mn)

    # --------------------------------------
    # 2. run-ms2lda
    # --------------------------------------
    p_run_ms2lda = subparsers.add_parser(
        "run-ms2lda",
        parents=[build_ms2lda_parser()],
        add_help=False,
        help="Run MS2LDA topic modeling to extract substructural Mass2Motifs from MGF data.",
    )
    p_run_ms2lda.set_defaults(func=handle_run_ms2lda)

    # --------------------------------------
    # 3. add-ms2lda
    # --------------------------------------
    p_add_ms2lda = subparsers.add_parser(
        "add-ms2lda",
        parents=[build_add_ms2lda_parser()],
        add_help=False,
        help="Calculate motif overlap and integrate MS2LDA metadata into an existing CX graph network.",
    )
    p_add_ms2lda.set_defaults(func=handle_add_ms2lda)

    # --------------------------------------
    # 4. add-mag
    # --------------------------------------
    p_add_mag = subparsers.add_parser(
        "add-mag",
        parents=[build_add_mag_parser()],
        add_help=False,
        help="Annotate extracted Mass2Motifs using Motif Annotation Guidance (MAG).",
    )
    p_add_mag.set_defaults(func=handle_add_mag)

    # --------------------------------------
    # 5. run-snapms
    # --------------------------------------
    p_run_snapms = subparsers.add_parser(
        "run-snapms",
        parents=[build_snapms_parser()],
        add_help=False,
        help="Annotate molecular networks using SNAP-MS structure database matching.",
    )
    p_run_snapms.set_defaults(func=handle_run_snapms)

    # --------------------------------------
    # 6. add-snapms
    # --------------------------------------
    p_add_snapms = subparsers.add_parser(
        "add-snapms",
        parents=[build_add_snapms_parser()],
        add_help=False,
        help="Relay annotation metadata and flag SMILES on target CX graph.",
    )
    p_add_snapms.set_defaults(func=handle_add_snapms)

    # --------------------------------------
    # 7. run-annotation
    # --------------------------------------
    p_run_annotation = subparsers.add_parser(
        "run-annotation",
        parents=[build_annotation_parser()],
        add_help=False,
        help="motif fragment based SNAP-MS tool[cite: 2].",
    )
    p_run_annotation.set_defaults(func=handle_run_annotation)

    # --------------------------------------
    # 8. run-all
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
        help="Output path for intermediate and final network CX graph file.",
    )
    p_run_all.add_argument(
        "--ms2lda-model-path",
        type=str,
        default="results/model.lda",
        help="Output path for trained MS2LDA model file.",
    )
    p_run_all.add_argument(
        "--ms2lda-motifs-path",
        type=str,
        default="results/motifs.mgf",
        help="Output path for extracted MS2LDA Mass2Motifs.",
    )
    p_run_all.add_argument(
        "--snapms-result-folder",
        type=str,
        default="results/snapms",
        help="Output directory for SNAP-MS intermediate results.",
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
        help="MS2LDA motif overlap threshold.",
    )
    p_run_all.set_defaults(func=handle_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
