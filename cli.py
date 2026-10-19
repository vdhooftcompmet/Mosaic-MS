import argparse
from pathlib import Path

from src.add_ms2lda.main import build_parser as build_add_ms2lda_parser
from src.add_ms2lda.main import main as add_ms2lda_main
from src.add_snapms.main import build_parser as build_add_snapms_parser
from src.add_snapms.main import main as add_snapms_main
from src.annotation.main import build_parser as build_annotation_parser
from src.annotation.main import main as run_annotation_main
from src.merge.main import build_parser as build_merge_parser
from src.merge.main import main as run_merge_main
from src.mn.main import build_parser as build_mn_parser
from src.mn.main import main as run_mn_main
from src.ms2lda.mag_main import build_parser as build_add_mag_parser
from src.ms2lda.mag_main import main as add_mag_main
from src.ms2lda.main import build_parser as build_ms2lda_parser
from src.ms2lda.main import main as run_ms2lda_main
from src.snapms.main import build_parser as build_snapms_parser
from src.snapms.main import main as run_snapms_main
from src.utils.folders import prepare_directory


def handle_run_mn(args: argparse.Namespace) -> None:
    """Executes full Molecular Network (MN) generation workflow."""
    run_mn_main(args)


def handle_run_ms2lda(args: argparse.Namespace) -> None:
    """Executes full MS2LDA workflow."""
    run_ms2lda_main(args)


def handle_add_ms2lda(args: argparse.Namespace) -> None:
    """Integrates motif detection and metadata injection into target CX graph."""
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
    """Appends SnapMS annotations to a CX graph and flags SMILES."""
    add_snapms_main(args)


def handle_run_annotation(args: argparse.Namespace) -> None:
    """Executes motif fragment based SNAP-MS workflow."""
    run_annotation_main(args)


def handle_run_merge(args: argparse.Namespace) -> None:
    """Merges Mass2Motifs across multiple models or files."""
    run_merge_main(args)


def handle_run_all(args: argparse.Namespace) -> None:
    """Executes the updated parallel Mosaic-MS pipeline sequentially/concurrently."""

    dir = Path("run_all")
    if not dir.exists():
        prepare_directory(dir, overwrite=True)

    # get input
    mgf = args.mgf

    # define file names
    base_graph = str(dir / "base.cx")
    threshold_graph = str(dir / "threshold.cx")
    rescued_graph = str(dir / "rescued.cx")

    print("  -> Building Molecular Network...")
    skip_mn = Path(base_graph).exists() and Path(threshold_graph).exists() and Path(rescued_graph).exists()
    if not skip_mn:
        handle_run_mn(
            argparse.Namespace(
                mgf=mgf,
                base_graph_path=base_graph,
                threshold_graph_path=threshold_graph,
                rescued_graph_path=rescued_graph,
            )
        )

    # define file names
    k_values = (
        [int(k) for k in args.k_values.split(",")]
        if isinstance(args.k_values, str)
        else args.k_values
    )
    motif_paths = [str(dir / f"motifs_k{k}.mgf") for k in k_values]
    model_paths = [str(dir / f"model_k{k}.lda") for k in k_values]


    print(f"  -> Running MS2LDA for multiple k values: {k_values}...")
    for k, motifs_p, model_p in zip(k_values, motif_paths, model_paths):
        skip_ms2lda = Path(motifs_p).exists() and Path(model_p).exists()
        if not skip_ms2lda:
            run_ms2lda_main(
                argparse.Namespace(
                    mgf=mgf,
                    model_path=model_p,
                    motifs_path=motifs_p,
                    nr_of_motifs=k,
                )
            )

    # define files
    merged_motifs = str(dir / "merged_motifs.mgf")
    merged_graph = str(dir / "merged_motifs.cx")

    print("  -> Merging Mass2Motifs...")
    skip_merge = Path(merged_motifs).exists() and Path(merged_graph).exists()
    if not skip_merge:
        run_merge_main(
            argparse.Namespace(
                motif_files=motif_paths,
                out_graph=merged_graph,
                out_motifs=merged_motifs,
            )
        )

    # define files
    annotation_results = str(dir / "motif_annotation")

    print("  -> Running Motif-based Annotation...")
    skip_annotation = Path(annotation_results).exists()
    if not skip_annotation:
        run_annotation_main(
            argparse.Namespace(
                mgf=mgf,
                motifs=merged_motifs,
                result_folder=annotation_results,
            )
        )
    # define files
    graphs = [base_graph, threshold_graph, rescued_graph]
    results = [
        str(dir / "snapms_base"),
        str(dir / "snapms_threshold"),
        str(dir / "snapms_rescued"),
    ]

    print("  -> Running SNAP-MS on Molecular Network...")
    for graph, res in zip(graphs, results):
        skip_snapms = Path(res).exists()
        if not skip_snapms:
            handle_run_snapms(
                argparse.Namespace(
                    graph=graph,
                    result_folder=res,
                )
            )

    print("  -> Adding Mass2Motifs to Molecular Network...")
    for graph in graphs:
        handle_add_ms2lda(
            argparse.Namespace(
                graph=graph,
                motifs=merged_motifs,
                threshold=0.01
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

    # 1. run-mn
    p_run_mn = subparsers.add_parser(
        "run-mn",
        parents=[build_mn_parser()],
        add_help=False,
        help="Build a molecular network from MGF spectrum data using SpecReBoot.",
    )
    p_run_mn.set_defaults(func=handle_run_mn)

    # 2. run-ms2lda
    p_run_ms2lda = subparsers.add_parser(
        "run-ms2lda",
        parents=[build_ms2lda_parser()],
        add_help=False,
        help="Run MS2LDA topic modeling to extract substructural Mass2Motifs from MGF data.",
    )
    p_run_ms2lda.set_defaults(func=handle_run_ms2lda)

    # 3. add-ms2lda
    p_add_ms2lda = subparsers.add_parser(
        "add-ms2lda",
        parents=[build_add_ms2lda_parser()],
        add_help=False,
        help="Calculate motif overlap and integrate MS2LDA metadata into an existing CX graph network.",
    )
    p_add_ms2lda.set_defaults(func=handle_add_ms2lda)

    # 4. add-mag
    p_add_mag = subparsers.add_parser(
        "add-mag",
        parents=[build_add_mag_parser()],
        add_help=False,
        help="Annotate extracted Mass2Motifs using Motif Annotation Guidance (MAG).",
    )
    p_add_mag.set_defaults(func=handle_add_mag)

    # 5. run-snapms
    p_run_snapms = subparsers.add_parser(
        "run-snapms",
        parents=[build_snapms_parser()],
        add_help=False,
        help="Annotate molecular networks using SNAP-MS structure database matching.",
    )
    p_run_snapms.set_defaults(func=handle_run_snapms)

    # 6. add-snapms
    p_add_snapms = subparsers.add_parser(
        "add-snapms",
        parents=[build_add_snapms_parser()],
        add_help=False,
        help="Relay annotation metadata and flag SMILES on target CX graph.",
    )
    p_add_snapms.set_defaults(func=handle_add_snapms)

    # 7. run-annotation
    p_run_annotation = subparsers.add_parser(
        "run-annotation",
        parents=[build_annotation_parser()],
        add_help=False,
        help="motif fragment based SNAP-MS tool.",
    )
    p_run_annotation.set_defaults(func=handle_run_annotation)

    # 8. run-merge
    p_run_merge = subparsers.add_parser(
        "run-merge",
        parents=[build_merge_parser()],
        add_help=False,
        help="Merge Mass2Motifs across multiple models or files.",
    )
    p_run_merge.set_defaults(func=handle_run_merge)

    # 9. run-all
    p_run_all = subparsers.add_parser(
        "run-all",
        help="Execute the full end-to-end Mosaic-MS pipeline sequentially/concurrently.",
    )
    p_run_all.add_argument(
        "--mgf", "-m", type=str, required=True, help="Path to input MGF spectrum file."
    )
    p_run_all.add_argument(
        "--k-values",
        type=str,
        required=True,
        help="Comma-separated list of k values for multi-k MS2LDA runs.",
    )
    p_run_all.set_defaults(func=handle_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
