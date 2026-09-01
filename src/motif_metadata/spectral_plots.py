import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import argparse

from pathlib import Path
from typing import Generator

from utils.cli import print_params , add_defaults


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defaults", type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--library", type=str)
    parser.add_argument("--query", type=str)
    
    parser.add_argument("--result-folder", type=str)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-batch-size", type=int, default=200)

    parser.add_argument("--similarity-threshold", type=float)

    parser.add_argument("--similarity-type", type=str)
    parser.add_argument("--flash-tolerance", type=float)
    parser.add_argument("--ms2deepscore-model-path", type=str)
    parser.add_argument("--spec2vec-model-path", type=str)

    parser.add_argument("--lda-model-path", type=str)
    parser.add_argument("--dataset-acquisition-type", type=str)
    parser.add_argument("--dataset-significant-digits", type=int)
    return parser


def derive_file_names(params) -> None:
    missing = (lambda p: not hasattr(params, p) or not getattr(params, p))

    if missing("folder"):
        params.folder = "."

    if missing("result_folder"):
        params.result_folder = f"{Path(params.library).stem}_motif_plots"

    if not Path(params.library).is_absolute():
        params.library = str( Path(params.folder) / params.library)
    
    if not Path(params.query).is_absolute():
        params.query = str( Path(params.folder) / params.query)

    if not Path(params.result_folder).is_absolute():
        params.result_folder = str( Path(params.folder) / params.result_folder)


if __name__ == "__main__":
    parser = cli()
    params = parser.parse_args()
    add_defaults(params, params.defaults)
    delattr(params, "defaults")
    derive_file_names(params)
    print("> running motif plotting with parameters:\n")
    print_params(params)


import logging

import tomotopy as tp
import gensim

from matchms import Spectrum
from matchms.importing import load_from_mgf
from matchms.similarity.FlashSimilarity import FlashSimilarity
from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore
from spec2vec import Spec2Vec
from unittest.mock import patch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from utils.folders import prepare_directory
from utils.constants import *
from utils.plots import *
from utils.images import *
from motif_metadata.motif_detection import run_overlap_scores_calculation


SCORE_KEY = "match_score"


def main(params):
    
    if not Path(params.result_folder).exists():
        prepare_directory(params.result_folder)

    with patch.object(logging.getLogger("matchms"), "level", logging.ERROR):
        query = list(load_from_mgf(params.query))

    library: Generator = load_from_mgf(params.library)

    similarity_function = get_similarity_function(params)
    library_matches = batch_wise_similarity(query, library, similarity_function, params)

    for query_index, matches in sorted(library_matches.items()):
        ranked_matches = sorted(matches, reverse=True, key=lambda x: x.get(SCORE_KEY))

        top_n = int(params.top_n)
        top_n_matches = ranked_matches[:top_n]

        query_spectra = query[query_index: query_index+1]

        scale = 1.6
        metadata_keys = [KEY_SPECTRUM_ID, SCORE_KEY, KEY_PRECURSOR_MZ, KEY_ADDUCT, "ion"]

        svg = plot_spectra(query_spectra, top_n_matches, metadata_keys, scale)

        save_path = Path(params.result_folder) / f"query_{query_index}_matches.svg"
        save_svg(svg, save_path)
  

def plot_spectra(reference: list[Spectrum], spectra: list[Spectrum], metadata_keys: list[str], scale: float):
    all_spectra = reference + spectra
    
    structure_svgs = [smiles_to_image(s.get("smiles"), scale) for s in all_spectra]
    
    axes = comparison_plot(reference, spectra, scale)
    ax_svgs = [ax_to_image(ax) for ax in axes]

    metadata_text = [spectrum_metadata_to_text(s, metadata_keys) for s in all_spectra]
    metadata_svgs = [text_to_image(t, scale) for t in metadata_text]

    columns = [
        list(ax_svgs) if not isinstance(ax_svgs, str) else [ax_svgs],
        list(structure_svgs) if not isinstance(structure_svgs, str) else [structure_svgs],
        list(metadata_svgs) if not isinstance(metadata_svgs, str) else [metadata_svgs]
    ]
    
    svg = make_vector_grid(columns)
    return svg


def spectrum_metadata_to_text(spectrum: Spectrum, keys: list[str | int]) -> str:
    text_rows = []
    for k in keys:
        text_rows.append(f"{k}:")
        text_rows.append(f"{spectrum.get(k)}")

    text = "\n".join(text_rows)
    return text


def batch_wise_similarity(query: list[Spectrum], library: Generator[Spectrum, None, None], similarity_function, params):
    library_matches = defaultdict(list)

    for batch in tqdm(batch_generator(params.max_batch_size, library)):

        for query_index, score, match in match_generator(query, batch, similarity_function):

            match.set(SCORE_KEY, score)
            library_matches[query_index].append(match)

    return library_matches


def batch_generator(max_batch_size: int, spectrum_generator):
    batch = []
    batch_size = 0

    for spectrum in spectrum_generator:

        batch.append(spectrum)
        batch_size += 1

        if batch_size < max_batch_size:
            continue

        yield batch

        batch = []
        batch_size = 0

    yield batch


def match_generator(query, batch, similarity_function):
    similarity = similarity_function(query, batch)
    threshold = float(params.similarity_threshold)

    assert len(query), len(batch) == similarity.shape

    for query_index in range(similarity.shape[0]):

        for spectrum_index, spectrum in enumerate(batch):
            score = similarity[query_index, spectrum_index]
            if score >= threshold:
                yield query_index, score, spectrum.clone()


def get_similarity_function(params):
    match params.similarity_type:
        case "cos":
            return get_cos_similarity_function(params)
        case "modcos":
            return get_modcos_similarity_function(params)
        case "spec2vec":
            return get_spec2vecs_similarity_function(params)
        case "ms2deepscore":
            return get_ms2deepscore_similarity_function(params)
        case "overlap":
            return get_overlap_similarity_function(params)
        case _:
            raise ValueError()


def get_cos_similarity_function(params):
    cos_similarity = FlashSimilarity(score_type="cosine", matching_mode="fragment", tolerance=params.flash_tolerance)

    def inner(query, batch):
        return cos_similarity.matrix(query, batch, "numpy")
    return inner


def get_modcos_similarity_function(params):
    modcos_similarity = FlashSimilarity(score_type="cosine", matching_mode="hybrid", tolerance=params.flash_tolerance)

    def inner(query, batch):
        return modcos_similarity.matrix(query, batch, "numpy")
    return inner


def get_spec2vecs_similarity_function(params):
    if not Path(params.spec2vec_model_path).exists():
        raise FileNotFoundError(f"file {params.spec2vec_model_path} not found")
    
    w2v = gensim.models.Word2Vec.load(str(params.spec2vec_model_path))
    spec2vec_similarity = Spec2Vec(model=w2v, intensity_weighting_power=0.5, allowed_missing_percentage=50, progress_bar=False)

    def inner(query, batch):
        return spec2vec_similarity.matrix(query, batch, "numpy")
    return inner


def get_ms2deepscore_similarity_function(params):
    if not Path(params.ms2deepscore_model_path).exists():
        raise FileNotFoundError(f"file {params.ms2deepscore_model_path} not found")
    
    ms2dp_model = load_model(str(params.ms2deepscore_model_path))
    ms2deepscore_similarity = MS2DeepScore(ms2dp_model, progress_bar=False)
    
    def inner(query, batch):
        return ms2deepscore_similarity.matrix(query, batch, "numpy")
    return inner


def get_overlap_similarity_function(params):
    if not Path(params.lda_model_path).exists():
        raise FileNotFoundError(f"file {params.lda_model_path} not found")
    
    lda_model = tp.LDAModel.load( str(params.lda_model_path) )

    def inner(query, batch):
        assert len(query) == lda_model.k, "imput query (motifs) do not match the nr of motifs of the model"
        beta_matrix, phi_matrix, theta_matrix, overlap_scores = run_overlap_scores_calculation(batch, lda_model, params)
        return overlap_scores
    return inner


if __name__ == "__main__":
    main(params)