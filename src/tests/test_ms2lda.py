import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tomotopy as tp
from matchms import Spectrum
from pathlib import Path

# Get the absolute path to 'src' directory (up 2 levels from src/test/my_file.py)
src_dir = Path(__file__).resolve().parent.parent

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ms2lda.ms2lda import (
    ConvergenceResult,
    _calculate_document_entropy,
    _calculate_topic_entropy,
    _extract_motif,
    _has_model_converged,
    extract_motifs,
    load_model,
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
    config.dataset_significant_digits = 2
    config.dataset_charge = 1
    return config


@pytest.fixture
def dummy_documents():
    """Provides sample documents matching expected feature prefixes."""
    return [
        ["frag@100.123", "loss@50.456", "frag@200.000"],
        ["frag@100.123", "frag@150.500", "loss@25.100"],
    ]


# --- Helper Function Tests ---

def test_calculate_document_entropy():
    # Setup mock tomotopy model with mock docs
    mock_model = MagicMock(spec=tp.LDAModel)
    doc1 = MagicMock()
    doc1.get_topic_dist.return_value = [0.5, 0.5]
    doc2 = MagicMock()
    doc2.get_topic_dist.return_value = [1.0, 0.0]  # Contains 0 to test 1e-12 replace
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

    # Relative changes: (100-99)/100 = 0.01, (99-98.5)/99 ≈ 0.005 -> Both < 0.1
    history = ConvergenceResult([100.0, 99.0, 98.5], [], [], [])
    
    assert _has_model_converged(history, mock_config) is True


def test_has_model_converged_false_insufficient_history(mock_config):
    mock_config.conv_window_size = 3
    history = ConvergenceResult([100.0, 99.0], [], [], [])
    
    assert _has_model_converged(history, mock_config) is False


def test_has_model_converged_false_above_threshold(mock_config):
    mock_config.conv_type = "perplexity_history"
    mock_config.conv_window_size = 2
    mock_config.conv_threshold = 0.01

    # Relative change: (100-90)/100 = 0.1 -> Exceeds 0.01 threshold
    history = ConvergenceResult([100.0, 95.0, 90.0], [], [], [])
    
    assert _has_model_converged(history, mock_config) is False


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
    k = 0

    spectrum = _extract_motif(k, topic, mock_config)

    assert isinstance(spectrum, Spectrum)
    # loss@50.222 -> -50.22, frag@100.555 -> 100.56 (sorted order)
    np.testing.assert_array_almost_equal(spectrum.peaks.mz, [-50.22, 100.56])
    # Intensities normalized relative to max (10.0 -> 1.0, 5.0 -> 0.5)
    np.testing.assert_array_almost_equal(spectrum.peaks.intensities, [0.5, 1.0])
    assert spectrum.get("id") == "motif_0"


def test_extract_motif_invalid_prefix(mock_config):
    topic = [("unknown@100.0", 1.0)]
    
    with pytest.raises(ValueError, match="invalid feature prefix"):
        _extract_motif(0, topic, mock_config)


def test_extract_motif_empty(mock_config):
    spectrum = _extract_motif(0, [], mock_config)
    
    assert isinstance(spectrum, Spectrum)
    assert len(spectrum.peaks.mz) == 0


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

def test_store_model_and_load_model(tmp_path):
    model_path = tmp_path / "lda_test.bin"
    
    # Create and save a minimal model
    mdl = tp.LDAModel(k=2)
    mdl.add_doc(["word1", "word2"])
    mdl.train(1)
    
    store_model(mdl, model_path)
    assert model_path.exists()

    loaded_mdl = load_model(model_path)
    assert isinstance(loaded_mdl, tp.LDAModel)
    assert loaded_mdl.k == 2


@patch("ms2lda.ms2lda.save_as_mgf")
def test_store_mass2motifs(mock_save_mgf, tmp_path):
    file_path = tmp_path / "motifs.mgf"
    motifs = [Spectrum(np.array([100.0]), np.array([1.0]))]

    store_mass2motifs(motifs, file_path)

    mock_save_mgf.assert_called_once_with(motifs, str(file_path), file_mode="w")
