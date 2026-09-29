from pathlib import Path

from matchms.importing import load_from_mgf

from src.ms2lda.ms2lda import (
    clean_spectra,
    extract_motifs,
    spectra_to_documents,
    store_mass2motifs,
    store_model,
    train_model,
)
from src.utils.configs import MS2LDAConfig


def main(config: MS2LDAConfig) -> None:

    spectra = load_from_mgf(config.mgf)
    cleaned_spectra = clean_spectra(spectra, config)
    spectral_docs = spectra_to_documents(cleaned_spectra, config)

    model, _convergence_result = train_model(spectral_docs, config)

    store_model(model, config.model_path)

    mass2motifs = extract_motifs(model, config)
    motifs_path = Path(config.motifs_path)
    store_mass2motifs(mass2motifs, motifs_path)
