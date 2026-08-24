import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pathlib import Path

from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf

from ms2lda.lda import train_model, extract_motifs
from ms2lda.preprocessing import clean_spectra, spectra_to_documents
from ms2lda.storage import store_model, store_mass2motifs

from utils.folders import save_config, prepare_directory


def main(params):

    if not Path(params.folder).exists():
        prepare_directory(params.folder)

    print("> preprocessing...")
    print("> loading .mgf...")

    cleaned_path = Path(params.mgf_cleaned)

    if cleaned_path.exists():
        cleaned_spectra = load_from_mgf(params.mgf_cleaned)

    else:
        spectra = load_from_mgf(params.mgf)

        print("> cleaning spectra...")
        cleaned_spectra = clean_spectra(spectra, params)
        
        print("> saving cleaned spectral data...")
        cleaned_path = Path(params.mgf_cleaned)
        save_as_mgf(cleaned_spectra, str(cleaned_path), file_mode="w")

    print("> converting to spectral docs...")
    spectral_docs = spectra_to_documents(cleaned_spectra, params)

    print("> training model...")
    model, convergence_result = train_model(spectral_docs, params)

    print("> storing model...")
    store_model(model, params.model_path)

    print("> extracting motifs...")
    mass2motifs = extract_motifs(model, params)

    print("> storing motifs...")
    motifs_path = Path(params.motifs_path)
    store_mass2motifs(mass2motifs, motifs_path)

    print(f"> results saved at {Path(params.folder)}...")