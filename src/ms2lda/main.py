from pathlib import Path
from matchms.importing import load_from_mgf
from ms2lda.lda import train_model, extract_motifs
from ms2lda.preprocessing import clean_spectra, spectra_to_documents
from ms2lda.storage import store_model, store_mass2motifs
from utils.configs import MS2LDAConfig


def main(config: MS2LDAConfig):

    spectra = load_from_mgf(config.mgf)
    cleaned_spectra = clean_spectra(spectra, config)
    spectral_docs = spectra_to_documents(cleaned_spectra, config)
    
    model, _convergence_result = train_model(spectral_docs, config)
    
    store_model(model, config.model_path)
    
    mass2motifs = extract_motifs(model, config)
    motifs_path = Path(config.motifs_path)
    store_mass2motifs(mass2motifs, motifs_path)