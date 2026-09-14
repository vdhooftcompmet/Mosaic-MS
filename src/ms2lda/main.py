import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pathlib import Path

from matchms.importing import load_from_mgf

from ms2lda.lda import train_model, extract_motifs
from ms2lda.preprocessing import clean_spectra, spectra_to_documents
from ms2lda.storage import store_model, store_mass2motifs

from utils.folders import save_config, prepare_directory


def main(params):

    spectra = load_from_mgf(params.mgf)
    cleaned_spectra = clean_spectra(spectra, params)
    spectral_docs = spectra_to_documents(cleaned_spectra, params)
    
    model, _convergence_result = train_model(spectral_docs, params)
    
    store_model(model, params.model_path)
    
    mass2motifs = extract_motifs(model, params)
    motifs_path = Path(params.motifs_path)
    store_mass2motifs(mass2motifs, motifs_path)