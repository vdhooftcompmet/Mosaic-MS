import numpy as np
import tomotopy as tp
from tqdm import tqdm
from matchms import Spectrum
import warnings
from collections import namedtuple
import logging
from unittest.mock import patch
from utils.configs import MS2LDAConfig


ConvergenceResult = namedtuple("convergence_result", ["perplexity_history", "log_likelihood_history", "entropy_history_doc", "entropy_history_topic"])


def train_model(documents: list[list[str]], config: MS2LDAConfig) -> tuple[tp.LDAModel, ConvergenceResult]:

    result = ConvergenceResult([], [], [], [])

    model_parameters = dict(
        rm_top=config.model_rm_top,
        min_cf=config.model_min_cf,
        min_df=config.model_min_df,
        alpha =config.model_alpha,
        eta   =config.model_eta,
        seed  =config.model_seed,
    )
    train_parameters = dict(
        parallel=config.train_parallel,
        workers =config.train_workers
    )

    model = tp.LDAModel(k=config.nr_of_motifs, **model_parameters)
    for document in documents:
        model.add_doc(document)

    warnings.filterwarnings("ignore", message="The training result may differ even with fixed seed if `workers` != 1.", category=RuntimeWarning)

    for i in tqdm(range(0, config.iterations, config.conv_step_size)):
        model.train(config.conv_step_size, **train_parameters)  # model is doing x amount (step size) of iterations

        result.perplexity_history     .append(model.perplexity)
        result.log_likelihood_history .append(model.ll_per_word)
        result.entropy_history_doc    .append(_calculate_document_entropy(model))
        result.entropy_history_topic  .append(_calculate_topic_entropy(model))

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


def _has_model_converged(convergence_history: ConvergenceResult, config: MS2LDAConfig) -> bool:
    convengence_type = config.conv_type
    history = getattr(convergence_history, convengence_type)
    window_size = config.conv_window_size
    epsilon = config.conv_threshold

    if len(history) <= window_size:
        return False

    changes = []
    for i in range(1, len(history)):
        change = abs(history[i] - history[i - 1]) / history[i - 1]
        changes.append(change)

    has_converged = all(change < epsilon for change in changes[-window_size:])
    if not has_converged:
        return False

    return True


def extract_motifs(model: tp.LDAModel, config: MS2LDAConfig) -> list[Spectrum]:

    result = []
    for k in range(model.k):
        topic = model.get_topic_words(k, top_n=config.top_n_words)

        if len(topic) == 0:
            print(" Warning !! topic with 0 features detected")

        with patch.object(logging.getLogger("matchms"), "level", logging.ERROR):
            mass2motif = _extract_motif(k, topic, config)
        if mass2motif is None:
            continue
        result.append(mass2motif)
    return result


def _extract_motif(k: int, topic: list[tuple[str, float]], config: MS2LDAConfig) -> Spectrum:
    significant_digits  = config.dataset_significant_digits
    charge              = config.dataset_charge

    fragments = []

    for motif_feature in topic:
        feature, importance = motif_feature
        importance = float(importance)

        is_loss, is_peak = feature.startswith("loss@"), feature.startswith("frag@")

        if not (is_loss or is_peak):
            raise ValueError(f"invalid feature prefix {feature}")

        if is_peak:
            mz_str: str = feature.removeprefix("frag@")
            mz: float = float(mz_str)
        if is_loss:
            mz_str: str = feature.removeprefix("loss@")
            mz: float = float(mz_str) * -1

        mz: float = round(mz, significant_digits)
        fragments.append((mz, importance))
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


