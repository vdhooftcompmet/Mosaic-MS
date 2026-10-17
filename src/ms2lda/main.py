import argparse
from pathlib import Path

from src.utils.cli import prepare_args
from src.utils.configs import MS2LDAConfig

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_MS2LDA_CONFIG = REPO / "config" / "ms2lda.yaml"


def main(args: argparse.Namespace) -> None:
    args = prepare_args(args, DEFAULT_MS2LDA_CONFIG)
    config = MS2LDAConfig(**dict(vars(args)))
    config.display()

    from matchms.importing import load_from_mgf

    from src.ms2lda.ms2lda import (
        clean_spectra,
        extract_motifs,
        spectra_to_documents,
        store_mass2motifs,
        store_model,
        train_model,
    )

    spectra = load_from_mgf(config.mgf)
    cleaned_spectra = clean_spectra(spectra, config)
    spectral_docs = spectra_to_documents(
        cleaned_spectra,
        config.dataset_acquisition_type == "DIA",
        config.dataset_significant_digits,
    )

    model, _convergence_result = train_model(spectral_docs, config)

    store_model(model, config.model_path)

    mass2motifs = extract_motifs(
        model,
        config.top_n_words,
        config.dataset_charge,
        config.dataset_significant_digits,
    )
    motifs_path = Path(config.motifs_path)
    store_mass2motifs(mass2motifs, motifs_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI tool for strata-ms workflows.")
    parser.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--mgf", "-m", type=str, default=None, help="Path to MGF spectrum file."
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Save path for trained LDA model."
    )
    parser.add_argument(
        "--motifs-path",
        type=str,
        default=None,
        help="Save path for Mass2Motifs (.mgf).",
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Max LDA training iterations."
    )
    parser.add_argument(
        "--nr-of-motifs",
        type=int,
        default=None,
        help="Total number of motifs to extract.",
    )
    parser.add_argument(
        "--top-n-words",
        type=int,
        default=None,
        help="Number of top fragment/loss features per motif.",
    )
    parser.add_argument(
        "--dataset-acquisition-type",
        choices=["DDA", "DIA"],
        default=None,
        help="Acquisition type mode.",
    )
    parser.add_argument(
        "--dataset-charge",
        type=int,
        choices=[1, -1],
        default=None,
        help="Ionization charge polarity.",
    )
    parser.add_argument(
        "--dataset-significant-digits",
        type=int,
        default=None,
        help="Precision for binning peak m/z values.",
    )
    parser.add_argument(
        "--train-parallel",
        type=int,
        default=None,
        help="Parallel threads for training.",
    )
    parser.add_argument(
        "--train-workers", type=int, default=None, help="Worker processes for training."
    )
    parser.add_argument(
        "--model-rm_top",
        type=int,
        default=None,
        help="Number of top global words to exclude.",
    )
    parser.add_argument(
        "--model-min-cf", type=int, default=None, help="Minimum feature count filter."
    )
    parser.add_argument(
        "--model-min-df",
        type=int,
        default=None,
        help="Minimum document frequency filter.",
    )
    parser.add_argument(
        "--model-alpha",
        type=float,
        default=None,
        help="Alpha hyperparameter for Dirichlet distribution.",
    )
    parser.add_argument(
        "--model-eta",
        type=float,
        default=None,
        help="Eta hyperparameter for Dirichlet distribution.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Random seed for LDA model initialization.",
    )
    parser.add_argument(
        "--conv-step-size",
        type=int,
        default=None,
        help="Frequency evaluating convergence.",
    )
    parser.add_argument(
        "--conv-window-size",
        type=int,
        default=None,
        help="Window size for convergence trends.",
    )
    parser.add_argument(
        "--conv-threshold",
        type=float,
        default=None,
        help="Target change threshold for convergence.",
    )
    parser.add_argument(
        "--conv-type", type=str, default=None, help="Convergence measurement strategy."
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
