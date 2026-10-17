import logging
import warnings
from argparse import Namespace
from collections import namedtuple
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import matchms.filtering as msfilters
import numpy as np
import tomotopy as tp
from matchms import Spectrum
from matchms.exporting import save_as_mgf

from src.utils.configs import MS2LDAConfig
from src.utils.motifs import (
    derive_dataset_acquisition_type,
    derive_significant_digits,
    get_topic_words,
    spectra_to_documents,
)
from src.utils.progress_bar import track


def clean_spectra(
    spectra: Generator[Spectrum, None, None] | list[Spectrum], config: MS2LDAConfig
) -> list[Spectrum]:
    result = []

    spectra = [x for x in spectra]

    for spectrum in spectra:
        spectrum = msfilters.default_filters(spectrum)
        spectrum = msfilters.add_retention_index(spectrum)
        spectrum = msfilters.add_retention_time(spectrum)
        spectrum = msfilters.normalize_intensities(spectrum)
        spectrum = msfilters.select_by_relative_intensity(
            spectrum,
            intensity_from=config.prep_min_intensity,
            intensity_to=config.prep_max_intensity,
        )
        spectrum = msfilters.select_by_mz(
            spectrum, mz_from=config.prep_min_mz, mz_to=config.prep_max_mz
        )
        spectrum = msfilters.reduce_to_number_of_peaks(
            spectrum, n_max=config.prep_max_frags
        )
        spectrum = msfilters.require_minimum_number_of_peaks(
            spectrum, n_required=config.prep_min_frags
        )

        if spectrum:
            result.append(spectrum)

    for i, spectrum in enumerate(result):
        spectrum.set("spectrum_id", i)

    return result


ConvergenceResult = namedtuple(
    "convergence_result",
    [
        "perplexity_history",
        "log_likelihood_history",
        "entropy_history_doc",
        "entropy_history_topic",
    ],
)


def train_model(
    documents: list[list[str]], config: MS2LDAConfig
) -> tuple[tp.LDAModel, ConvergenceResult]:

    result = ConvergenceResult([], [], [], [])

    model_parameters = dict(
        rm_top=config.model_rm_top,
        min_cf=config.model_min_cf,
        min_df=config.model_min_df,
        alpha=config.model_alpha,
        eta=config.model_eta,
        seed=config.model_seed,
    )
    train_parameters = dict(
        iterations=config.conv_step_size,
        workers=config.train_workers,
        parallel=config.train_parallel,
        freeze_topics=False,
    )

    model = tp.LDAModel(k=config.nr_of_motifs, **model_parameters)
    for document in documents:
        model.add_doc(document)

    warnings.filterwarnings(
        "ignore",
        message="The training result may differ even with fixed seed if `workers` != 1.",
        category=RuntimeWarning,
    )

    for i in track(
        range(0, config.iterations, config.conv_step_size),
        description="running ms2lda...",
    ):
        model.train(
            **train_parameters
        )  # model is doing x amount (step size) of iterations

        result.perplexity_history.append(model.perplexity)
        result.log_likelihood_history.append(model.ll_per_word)
        result.entropy_history_doc.append(_calculate_document_entropy(model))
        result.entropy_history_topic.append(_calculate_topic_entropy(model))

        if _has_model_converged(result, config):
            warnings.resetwarnings()
            return model, result

    print("> WARNING! - Model did not converge!!")
    return model, result


def _calculate_document_entropy(model: tp.LDAModel) -> float:
    entropy_values = []

    for doc in model.docs:
        topic_dist = doc.get_topic_dist()
        topic_dist = np.where(np.array(topic_dist) == 0, 1e-12, topic_dist)

        entropy = -np.sum(topic_dist * np.log(topic_dist))
        entropy_values.append(entropy)

    return np.mean(entropy_values)


def _calculate_topic_entropy(model: tp.LDAModel) -> float:
    entropy_values = []

    for k in range(model.k):
        word_dist = model.get_topic_word_dist(k)
        word_dist = np.where(np.array(word_dist) == 0, 1e-12, word_dist)

        entropy = -np.sum(word_dist * np.log(word_dist))
        entropy_values.append(entropy)

    return np.mean(entropy_values)


def _has_model_converged(
    convergence_history: ConvergenceResult, config: MS2LDAConfig
) -> bool:
    convengence_type = config.conv_type
    history = getattr(convergence_history, convengence_type)
    window_size = config.conv_window_size
    epsilon = config.conv_threshold

    if len(history) <= window_size:
        return False

    changes = []
    for i in range(1, len(history)):
        change = abs(history[i] - history[i - 1]) / (abs(history[i - 1]) + 1e-12)
        changes.append(change)

    has_converged = all(change < epsilon for change in changes[-window_size:])
    if not has_converged:
        return False

    return True


def extract_motifs(
    model: tp.LDAModel,
    top_n_words: int,
    charge: int,
    significant_digits: None | int = None,
) -> list[Spectrum]:
    topic_words = get_topic_words(model)

    if significant_digits is None:
        significant_digits = derive_significant_digits(topic_words)

    result = []
    for k in range(model.k):
        topic = model.get_topic_words(k, top_n=top_n_words)

        if len(topic) == 0:
            print(" Warning !! topic with 0 features detected")

        with patch.object(logging.getLogger("matchms"), "level", logging.ERROR):
            mass2motif = _extract_motif(k, topic, significant_digits, charge)
        if mass2motif is None:
            continue
        result.append(mass2motif)
    return result


def _extract_motif(
    k: int,
    topic: list[tuple[str, float]] | list[tuple[str, int, float]],
    significant_digits: int,
    charge: int,
) -> Spectrum:

    fragments: list[tuple[float, float]] = []

    for motif_feature in topic:
        feature = motif_feature[0]
        importance = motif_feature[-1]
        importance = float(importance)

        if feature.startswith("frag@"):
            feature_mz_str = feature.removeprefix("frag@")
            feature_mz = float(feature_mz_str)
        elif feature.startswith("loss@"):
            feature_mz_str = feature.removeprefix("loss@")
            feature_mz = float(feature_mz_str) * -1
        else:
            raise ValueError(f"invalid feature prefix {feature}")

        feature_mz: float = round(feature_mz, significant_digits)
        fragments.append((feature_mz, importance))
        continue

    fragments.sort()

    m2m_id = f"motif_{k}".strip()
    accuracy = (1 / (10**significant_digits)) / 2

    metadata = dict(id=m2m_id, charge=charge, accuracy=accuracy)

    if len(fragments) == 0:
        return Spectrum(np.array([]), np.array([]), metadata)

    mz = [x[0] for x in fragments]
    intensities = [x[1] for x in fragments]

    max_intensity = max(intensities)
    normalized_intensities = [intensity / max_intensity for intensity in intensities]

    return Spectrum(np.array(mz), np.array(normalized_intensities), metadata)


def store_mass2motifs(mass2motifs: list[Spectrum], path: str | Path) -> None:
    save_as_mgf(mass2motifs, str(path), file_mode="w")


def store_model(model: tp.LDAModel, path: Path | str) -> None:
    model.save(str(path))


def load_model(path: Path | str) -> tp.LDAModel:
    path = Path(path)
    model = tp.LDAModel.load(str(path))
    return model
