from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tomotopy as tp
from matchms import Spectrum
import matchms.filtering as msfilters

# Assuming ms2lda.py and utils/configs.py are in your python path
from ms2lda.ms2lda import (
    ConvergenceResult,
    _calculate_document_entropy,
    _calculate_topic_entropy,
    _extract_motif,
    _has_model_converged,
    clean_spectra,
    extract_motifs,
    load_model,
    spectra_to_documents,
    store_mass2motifs,
    store_model,
    train_model,
)
from utils.configs import MS2LDAConfig


# --- Fixtures ---

@pytest.fixture
def mock_config():
    """Provides a dummy MS2LDAConfig object with preset values."""
    config = MagicMock(spec=MS2LDAConfig)
    # Preprocessing configs
    config.prep_min_intensity = 0.01
    config.prep_max_intensity = 1.0
    config.prep_min_mz = 10.0
    config.prep_max_mz = 1000.0
    config.prep_min_frags = 1
    config.prep_max_frags = 100
    # Dataset configs
    config.dataset_acquisition_type = "DDA"
    config.dataset_significant_digits = 2
    config.dataset_charge = 1
    # Model configs
    config.model_rm_top = 0
    config.model_min_cf = 0
    config.model_min_df = 0
    config.model_alpha = 0.1
    config.model_eta = 0.01
    config.model_seed = 42
    config.train_parallel = 0
    config.train_workers = 1
    config.nr_of_motifs = 2
    config.iterations = 10
    config.conv_step_size = 5
    config.conv_type = "perplexity_history"
    config.conv_window_size = 2
    config.conv_threshold = 0.05
    config.top_n_words = 5
    return config


@pytest.fixture
def sample_spectrum():
    """Provides a basic matchms Spectrum object with computed losses."""
    mz = np.array([100.123, 200.456], dtype=float)
    intensities = np.array([0.5, 1.0], dtype=float)
    metadata = {"precursor_mz": 300.0, "charge": 1}

    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    # Correct matchms way to attach losses:
    return spectrum


@pytest.fixture
def dummy_documents():
    """Provides sample documents matching expected feature prefixes."""
    return [
        ["frag@100.12", "loss@50.46", "frag@200.00"],
        ["frag@100.12", "frag@150.50", "loss@25.10"],
    ]


# --- Tests for New Preprocessing & Document Functions ---

@patch("ms2lda.ms2lda.msfilters")
def test_clean_spectra(mock_msfilters, mock_config, sample_spectrum):
    # Setup mock return values for chained matchms filters
    mock_msfilters.default_filters.side_effect = lambda s: s
    mock_msfilters.add_retention_index.side_effect = lambda s: s
    mock_msfilters.add_retention_time.side_effect = lambda s: s
    mock_msfilters.normalize_intensities.side_effect = lambda s: s
    mock_msfilters.select_by_relative_intensity.side_effect = lambda s, **kw: s
    mock_msfilters.select_by_mz.side_effect = lambda s, **kw: s
    mock_msfilters.reduce_to_number_of_peaks.side_effect = lambda s, **kw: s
    mock_msfilters.require_minimum_number_of_peaks.side_effect = lambda s, **kw: s

    # Pass as generator
    spectra_gen = (s for s in [sample_spectrum])
    cleaned = clean_spectra(spectra_gen, mock_config)

    assert len(cleaned) == 1
    assert cleaned[0].get("spectrum_id") == 0


def test_spectra_to_documents_dda(mock_config, sample_spectrum):
    mock_config.dataset_acquisition_type = "DDA"

    documents = spectra_to_documents([sample_spectrum], mock_config)

    assert len(documents) == 1
    doc = documents[0]

    # Intensities (normalized relative to max: 0.5 -> 50 repetitions, 1.0 -> 100 repetitions)
    # Total count = 50 * frag + 100 * frag + 50 * loss + 100 * loss = 300 words
    assert len(doc) == 300
    assert "frag@100.12" in doc
    assert "frag@200.46" in doc
    assert "loss@199.88" in doc
    assert "loss@99.54" in doc


def test_spectra_to_documents_dia_ignores_losses(mock_config, sample_spectrum):
    mock_config.dataset_acquisition_type = "DIA"

    documents = spectra_to_documents([sample_spectrum], mock_config)

    assert len(documents) == 1
    doc = documents[0]

    # DIA mode skips losses entirely
    # Total count = 50 * frag + 100 * frag = 150 words
    assert len(doc) == 150
    assert "frag@100.12" in doc
    assert not any(w.startswith("loss@") for w in doc)


def test_spectra_to_documents_invalid_acquisition_type(mock_config, sample_spectrum):
    mock_config.dataset_acquisition_type = "INVALID_TYPE"

    with pytest.raises(AssertionError):
        spectra_to_documents([sample_spectrum], mock_config)


# --- Entropy Calculation Tests ---

def test_calculate_document_entropy():
    mock_model = MagicMock(spec=tp.LDAModel)
    doc1 = MagicMock()
    doc1.get_topic_dist.return_value = [0.5, 0.5]
    doc2 = MagicMock()
    doc2.get_topic_dist.return_value = [1.0, 0.0]  # Tests 1e-12 replacement
    mock_model.docs = [doc1, doc2]

    entropy = _calculate_document_entropy(mock_model)

    assert isinstance(entropy, float)
    assert entropy > 0


def test_calculate_topic_entropy():
    mock_model = MagicMock(spec=tp.LDAModel)
    mock_model.k = 2
    mock_model.get_topic_word_dist.side_effect = [
        [0.5, 0.5],
        [1.0, 0.0],
    ]

    entropy = _calculate_topic_entropy(mock_model)

    assert isinstance(entropy, float)
    assert entropy > 0


# --- Convergence Tests ---

def test_has_model_converged_true(mock_config):
    mock_config.conv_type = "perplexity_history"
    mock_config.conv_window_size = 2
    mock_config.conv_threshold = 0.1

    history = ConvergenceResult([100.0, 99.0, 98.5], [], [], [])

    assert _has_model_converged(history, mock_config) is True


def test_has_model_converged_false_insufficient_history(mock_config):
    mock_config.conv_window_size = 3
    history = ConvergenceResult([100.0, 99.0], [], [], [])

    assert _has_model_converged(history, mock_config) is False


def test_has_model_converged_handles_zero_values(mock_config):
    mock_config.conv_type = "log_likelihood_history"
    mock_config.conv_window_size = 2
    mock_config.conv_threshold = 0.01

    # Regression test verifying zero values do not raise ZeroDivisionError
    history = ConvergenceResult([], [0.0, 0.0, 0.0], [], [])

    assert _has_model_converged(history, mock_config) is True


# --- Train Model Tests ---

def test_train_model_runs_and_converges(mock_config, dummy_documents):
    model, result = train_model(dummy_documents, mock_config)

    assert isinstance(model, tp.LDAModel)
    assert isinstance(result, ConvergenceResult)
    assert len(result.perplexity_history) > 0
    assert len(result.log_likelihood_history) == len(result.perplexity_history)


@patch("ms2lda.ms2lda._has_model_converged", return_value=False)
def test_train_model_warns_on_non_convergence(mock_has_converged, mock_config, dummy_documents, capsys):
    model, result = train_model(dummy_documents, mock_config)

    captured = capsys.readouterr()
    assert "> WARNING! - Model did not converge!!" in captured.out


# --- Motif Extraction Tests ---

def test_extract_motif_valid(mock_config):
    topic = [("frag@100.555", 10.0), ("loss@50.222", 5.0)]

    spectrum = _extract_motif(0, topic, mock_config)

    assert isinstance(spectrum, Spectrum)
    # loss@50.222 -> -50.22, frag@100.555 -> 100.56
    np.testing.assert_array_almost_equal(spectrum.peaks.mz, [-50.22, 100.56])
    np.testing.assert_array_almost_equal(spectrum.peaks.intensities, [0.5, 1.0])
    assert spectrum.get("id") == "motif_0"


def test_extract_motif_invalid_prefix(mock_config):
    topic = [("unknown@100.0", 1.0)]

    with pytest.raises(ValueError, match="invalid feature prefix"):
        _extract_motif(0, topic, mock_config)


def test_extract_motifs_integration(mock_config):
    mock_model = MagicMock(spec=tp.LDAModel)
    mock_model.k = 2
    mock_model.get_topic_words.side_effect = [
        [("frag@100.0", 1.0)],
        [("loss@20.0", 2.0)],
    ]

    motifs = extract_motifs(mock_model, mock_config)

    assert len(motifs) == 2
    assert all(isinstance(m, Spectrum) for m in motifs)


# --- File Persistence Tests ---

def test_store_model_and_load_model(tmp_path: Path):
    model_path = tmp_path / "lda_test.bin"

    mdl = tp.LDAModel(k=2)
    mdl.add_doc(["word1", "word2"])
    mdl.train(1)

    store_model(mdl, model_path)
    assert model_path.exists()

    loaded_mdl = load_model(model_path)
    assert isinstance(loaded_mdl, tp.LDAModel)
    assert loaded_mdl.k == 2


@patch("ms2lda.ms2lda.save_as_mgf")
def test_store_mass2motifs(mock_save_mgf, tmp_path: Path):
    file_path = tmp_path / "motifs.mgf"
    motifs = [Spectrum(np.array([100.0]), np.array([1.0]))]

    store_mass2motifs(motifs, file_path)

    mock_save_mgf.assert_called_once_with(motifs, str(file_path), file_mode="w")
