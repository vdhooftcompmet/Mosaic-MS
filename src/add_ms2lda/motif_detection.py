
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pathlib import Path

import argparse
from utils.folders import load_params
from utils.cli import print_params , add_defaults, completeness_check


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--mgf", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--result-path", type=str)
    parser.add_argument("--dataset-acquisition-type", type=str)
    parser.add_argument("--dataset-significant-digits", type=int)
    return parser

def derive_file_names(params) -> None:
    missing = (lambda p: not hasattr(params, p) or not getattr(params, p))

    if missing("folder"):
        params.folder = "."

    if missing("result_path"):
        params.result_path = f"{Path(params.mgf).stem}_motifs_detected.csv"

    if not Path(params.mgf).is_absolute():
        params.mgf = str( Path(params.folder) / params.mgf)
    
    if not Path(params.model_path).is_absolute():
        params.model_path = str( Path(params.folder) / params.model_path)

    if not Path(params.result_path).is_absolute():
        params.result_path = str( Path(params.folder) / params.result_path)


if __name__ == "__main__":
    parser = cli()
    params = parser.parse_args()
    add_defaults(params, params.defaults)
    delattr(params, "defaults")
    derive_file_names(params)
    completeness_check(params)
    print("> running motif detection with parameters:\n")
    print_params(params)


import numpy as np
from matchms import Spectrum
from argparse import Namespace
import tomotopy as tp
from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
from matchms.importing import load_from_mgf

from ms2lda.preprocessing import spectra_to_documents


def main(params) -> None:
    derive_file_names(params)

    spectra = [x for x in tqdm(load_from_mgf(params.mgf))]
    model = tp.LDAModel.load( str(params.model_path) )
    
    beta_matrix, phi_matrix, theta_matrix, overlap_scores = run_overlap_scores_calculation(spectra, model, params)

    np.savetxt(str(params.result_path), overlap_scores, delimiter=",")


def run_overlap_scores_calculation(spectra: List[Spectrum],  model: tp.LDAModel,  params: Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # see original/MS2LDA/Visualization/lda_dict for original function

    spectra = [s for s in spectra]

    n_words = len(model.used_vocabs)
    n_docs  = len(spectra)
    n_topics = model.k

    words_per_spectra = spectra_to_documents(spectra, params)

    documents = []
    for words in tqdm(words_per_spectra):
        words = list(words)

        if not words:
            documents.append(None)
        else:
            doc = model.make_doc(list(words))
            model.infer(doc)  
            documents.append(doc)

    beta_matrix = np.zeros((n_topics, n_words), dtype=float)  # Beta signifies the distribution of words for every topic
    for i in range(n_topics):
        word_probs = model.get_topic_word_dist(i)
        beta_matrix[i, :] = np.array(word_probs)

    theta_matrix = np.zeros((n_topics, n_docs))  # Theta signifies the presence of each topic for every doc
    for i, doc in enumerate(documents):

        if doc is None:
            continue

        theta_matrix[:, i] = doc.get_topic_dist() 

    word_counts = np.zeros((n_words, n_docs), dtype=np.int32)  # Word counts is a simple histogram that couns how mane times the same word occurs in the same doc
    for i, doc in enumerate(documents):

        if doc is None:
            continue

        doc_indices = np.array([w for w in doc.words], dtype=np.int32)
        valid_indices = doc_indices[doc_indices < n_words]
        
        np.add.at(word_counts[:, i], valid_indices, 1)

    phi_matrix = beta_matrix @ word_counts   # tw * wd -> td; Phi signifies the topic distribution in every doc 
    phi_matrix /= phi_matrix.sum(axis=0, keepdims=True) + 1e-12  # Phi is noralized for topic size to make a fair comparison

    overlap_scores = phi_matrix * theta_matrix

    return beta_matrix, phi_matrix, theta_matrix, overlap_scores


if __name__ == "__main__":
    main(params)
