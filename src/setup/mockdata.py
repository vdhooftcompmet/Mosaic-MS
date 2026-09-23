import random
import numpy as np
from tqdm import tqdm
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.filtering.default_pipelines import DEFAULT_FILTERS, CLEAN_PEAKS
from matchms.filtering.SpectrumProcessor import SpectrumProcessor
from matchms.similarity.FlashSimilarity import FlashSimilarity
from rdkit import Chem


def clean_mgf(input_mgf, output_mgf):

    spectra = list(load_from_mgf( str(input_mgf) ))
    spectrum_processor = SpectrumProcessor(DEFAULT_FILTERS + CLEAN_PEAKS)
    result, _ = spectrum_processor.process_spectra(spectra, progress_bar=True)
    result = list(result)

    for i, spectrum in enumerate(result):
        spectrum.set("spectrum_id", i)

    save_as_mgf(spectra, str(output_mgf), file_mode="w")


def filter_by_inchikey(input_mgf, output_mgf):
    result = []
    seen = set()

    spectra = list( load_from_mgf( str(input_mgf) ) )
    spectra = random.sample(spectra, len(spectra))

    for spectrum in tqdm(spectra):
        inchi_key = _smiles_to_inchikey(spectrum.get("smiles"))

        if inchi_key in seen:
            continue
        
        seen.add(inchi_key)
        result.append(spectrum)

    save_as_mgf(result, str(output_mgf), file_mode="w")


def filter_by_cosine(input_mgf, output_mgf):
    result = []
    skip = set() 

    spectra = list( load_from_mgf( str(input_mgf) ) )
    spectra = random.sample(spectra, len(spectra))

    similarity = FlashSimilarity(score_type="cosine", matching_mode="fragment", tolerance=0.01).matrix(spectra, spectra, is_symmetric=True)

    for i, spectrum in tqdm(enumerate(spectra)):
        if i in skip:
            continue

        similarities = similarity[i]
        similarities = similarities.flatten()
        near_replicates = np.where(similarities >= 0.99)[0]

        skip.update(near_replicates)

        result.append(spectrum)

    save_as_mgf(result, str(output_mgf), file_mode="w")


def _smiles_to_inchikey(smiles):
    if not smiles: 
        return None

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    Chem.RemoveStereochemistry(mol)
    inchi_key = Chem.MolToInchiKey(mol)
    return inchi_key
