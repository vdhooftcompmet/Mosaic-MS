import gensim
import numpy as np
from joblib import parallel_backend
from typing import Any, List
from argparse import Namespace
from tqdm import tqdm
from matchms import Spectrum
from pathlib import Path
from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore
from spec2vec import Spec2Vec
from matchms.similarity.FlashSimilarity import FlashSimilarity
from utils.context import suppress_output
from utils.constants import *

def run_bootstrap(
        spectra: list[Spectrum], 
        params: Namespace, 
        ms2deepscore_model_path=None, 
        spec2vec_model_path=None, 
):
    average_similarity, support = calculate_bootstrapping(spectra, params.similarity_type, params, ms2deepscore_model_path, spec2vec_model_path)
    return average_similarity, support


def plain_similarity(
        spectra: list[Spectrum], 
        params: Namespace | None = None, 
):
    similarity_metric = get_similarity(params.similarity_type, params.flash_tolerance, params.ms2deepscore_model_path, params.spec2vec_model_path)
    with suppress_output():
        similarity_matrix = similarity_metric.matrix(list(spectra), list(spectra), array_type="numpy", is_symmetric=True)

    return similarity_matrix


def calculate_bootstrapping(
        spectra: list[Spectrum], 
        params: Namespace, 
) -> tuple[np.ndarray, np.ndarray]:
    
    
    bins = global_bins(spectra, params.binning_decimals)
    binned_spectra = bin_spectra(spectra, params.binning_decimals)

    dataset_size = len(binned_spectra)
    similarity_metric = get_similarity(params.similarity_type, params.flash_tolerance, params.ms2deepscore_model_path, params.spec2vec_model_path)

    random_generator = np.random.default_rng(params.seed)

    total_pair_similarities = np.zeros((dataset_size, dataset_size), dtype=float)
    total_edge_support      = np.zeros((dataset_size, dataset_size), dtype=float)

    for b in tqdm(range(params.B)):

        masked_spectra = _mask_spectra_globally(random_generator, bins, binned_spectra)

        with parallel_backend("loky", n_jobs=params.n_jobs):
            with suppress_output():
                similarity_matrix = similarity_metric.matrix(masked_spectra, masked_spectra, array_type="numpy", is_symmetric=True)

        top_k_nearest_neighbours = mutual_topk(similarity_matrix, params.k)
        top_k_nearest_neighbours_binary = (top_k_nearest_neighbours != 0).astype(int)

        total_pair_similarities += similarity_matrix
        total_edge_support      += top_k_nearest_neighbours_binary
        

    mean_similarities = total_pair_similarities / params.B

    np.fill_diagonal(mean_similarities , 1)  # needed to exaxtly match original implementation
    mean_edge_support = total_edge_support / params.B
    return mean_similarities, mean_edge_support


def global_bins(spectra: List[Spectrum], decimals: int) -> np.ndarray[float]:
    all_binned_mz = []

    for spec in spectra:
        rounded_mz_values = np.round(spec.peaks.mz, decimals)
        
        for mz in rounded_mz_values:
            all_binned_mz.append(mz)

    unique_mz = set(all_binned_mz)
    sorted_mz = sorted(unique_mz)

    return np.asarray(sorted_mz)


def bin_spectra(spectra: List[Spectrum], decimals: int) -> List[Spectrum]:
    binned_spectra = []

    for spec in spectra:
        rounded_mz = np.round(spec.peaks.mz, decimals)
        intensities = spec.peaks.intensities.copy()
        metadata = spec.metadata.copy() if spec.metadata else None

        new_spectrum = Spectrum(rounded_mz,  intensities, metadata)
        binned_spectra.append(new_spectrum)

    return binned_spectra


def mutual_topk(A, k):
    n = A.shape[0]
    A_work = A.copy()
    np.fill_diagonal(A_work, -np.inf)

    row_sorted = np.argsort(A_work, axis=1)[:, ::-1]  # consider making this argpartition
    row_topk = row_sorted[:, :k]

    row_mask = np.zeros_like(A_work, dtype=bool)
    rows = np.arange(n)[:, None]
    row_mask[rows, row_topk] = True

    mutual_mask = row_mask & row_mask.T

    result = A.copy()
    result[~mutual_mask] = 0
    return result


def get_similarity(method_name: str, flash_tolerance: float, ms2deepscore_model_path=None, spec2vec_model_path=None):
    match method_name:
        case "cos" | "cosine":
            return FlashSimilarity(score_type="cosine", matching_mode="fragment", tolerance=flash_tolerance)
        
        case "modcos" | "modified_cosine":
            return FlashSimilarity(score_type="cosine", matching_mode="hybrid", tolerance=flash_tolerance)
        
        case "ms2ds" | "ms2dp" | "ms2deepscore":
            if not Path(ms2deepscore_model_path).exists():
                raise FileNotFoundError(f"file {ms2deepscore_model_path} not found")
            
            ms2dp_model = load_model(str(ms2deepscore_model_path))
            return MS2DeepScore(ms2dp_model, progress_bar=False)
        
        case "s2v" | "spec2vec":
            if not Path(spec2vec_model_path).exists():
                raise FileNotFoundError(f"file {spec2vec_model_path} not found")
            
            w2v = gensim.models.Word2Vec.load(str(spec2vec_model_path))
            return Spec2Vec(model=w2v, intensity_weighting_power=0.5, allowed_missing_percentage=5.0, progress_bar=False)
        
        case _:
            raise ValueError(f"unknown option {method_name}")


def _mask_spectra_globally(random_generator: Any, global_bins: np.ndarray, binned_spectra: np.ndarray) -> list[Spectrum]:
    result = []

    sampled_indices = random_generator.integers(0, len(global_bins), size=len(global_bins))

    sampled_bins = global_bins[sampled_indices]
    sampled_bins = np.unique(sampled_bins)

    for index, spectrum in enumerate(binned_spectra):
        mask = np.isin(spectrum.peaks.mz, sampled_bins)

        mz = spectrum.peaks.mz[mask] 
        intensities = spectrum.peaks.intensities[mask]

        mz = mz.astype("float32")
        intensities = intensities.astype("float32")

        if len(mz) == 0:
            mz = np.array([ global_bins[0] ], dtype="float32")
            intensities = np.array([0.0],     dtype="float32")

        masked_spectrum = Spectrum(mz, intensities, spectrum.metadata)
        result.append(masked_spectrum)

    return result
