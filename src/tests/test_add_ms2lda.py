import json
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="builtin type _VocabDict has no __module__ attribute",
)

from argparse import Namespace
from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
from matchms import Spectrum

from src.add_ms2lda.main import (
    derive_dataset_acquisition_type,
    derive_significant_digits,
    get_topic_words,
    main,
    parse_spectrum_peaks,
    run_overlap_scores_calculation,
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="builtin type _VocabDict has no __module__ attribute",
)

# --- Fixtures ---


@pytest.fixture
def mock_params():
    """Fixture providing CLI arguments as a Namespace object."""
    return Namespace(
        model="dummy_model.bin",
        graph="dummy_graph.cx2",
        threshold=0.2,
        binning_decimals=2,
    )


@pytest.fixture
def mock_graph():
    """Fixture providing a NetworkX graph populated with peak JSON metadata."""
    G = nx.Graph()
    G.add_node(
        0,
        peaks_json=json.dumps([[100.1, 10.0], [200.2, 50.0]]),
        rtinminutes=1.5,
        compound_name="Compound_1",
    )
    G.add_node(
        1,
        peaks_json=[[150.3, 20.0], [250.4, 80.0]],
        rtinminutes=2.5,
        compound_name="Compound_2",
    )
    return G


@pytest.fixture
def mock_lda_model():
    """Fixture providing a mocked tomotopy LDAModel."""
    model = MagicMock()
    model.k = 2  # 2 topics
    model.used_vocabs = ["loss@100.12", "fragment@200.45"]

    # Mock get_topic_words
    model.get_topic_words.side_effect = lambda topic_id, top_n=10: [
        ("loss@100.12", 0.5),
        ("fragment@200.45", 0.5),
    ]

    # Mock get_topic_word_dist
    model.get_topic_word_dist.return_value = [0.6, 0.4]

    # Mock document creation and inference
    mock_doc = MagicMock()
    mock_doc.get_topic_dist.return_value = np.array([0.7, 0.3])
    mock_doc.words = [0, 1]
    model.make_doc.return_value = mock_doc

    return model


# --- Helper Function Tests ---


@pytest.mark.parametrize(
    "input_data, expected",
    [
        ("[[100.1, 10.0], [200.2, 50.0]]", [[100.1, 10.0], [200.2, 50.0]]),
        ([[100.1, 10.0], [200.2, 50.0]], [[100.1, 10.0], [200.2, 50.0]]),
    ],
)
def test_parse_spectrum_peaks(input_data, expected) -> None:
    result = parse_spectrum_peaks(input_data)
    assert result == expected


def test_get_topic_words(mock_lda_model) -> None:
    words = get_topic_words(mock_lda_model)
    assert words == ["loss@100.12", "fragment@200.45", "loss@100.12", "fragment@200.45"]


@pytest.mark.parametrize(
    "words, expected_decimals",
    [
        (["loss@100.12", "fragment@200.456"], 3),
        (["loss@100.1", "fragment@200.4"], 1),
        (["word_without_at"], 0),
    ],
)
def test_derive_significant_digits(words: Sequence[str], expected_decimals) -> None:
    assert derive_significant_digits(words) == expected_decimals


@pytest.mark.parametrize(
    "words, expected_type",
    [
        (["loss@100.12", "fragment@200.45"], "DDA"),
        (["fragment@100.12", "fragment@200.45"], "DIA"),
    ],
)
def test_derive_dataset_acquisition_type(words: Sequence[str], expected_type) -> None:
    assert derive_dataset_acquisition_type(words) == expected_type


# --- Core Calculation Tests ---


@patch("src.add_ms2lda.main.spectra_to_documents")
def test_run_overlap_scores_calculation(
    mock_spectra_to_docs, mock_lda_model, mock_params
) -> None:
    spec1 = Spectrum(mz=np.array([100.1, 200.2]), intensities=np.array([10.0, 50.0]))
    spec2 = Spectrum(mz=np.array([150.3, 250.4]), intensities=np.array([20.0, 80.0]))
    spectra = [spec1, spec2]

    # Mock words generated per spectrum
    mock_spectra_to_docs.return_value = [["loss@100.12"], ["fragment@200.45"]]

    beta, phi, theta, overlap_scores = run_overlap_scores_calculation(
        spectra, mock_lda_model, mock_params
    )

    assert beta.shape == (2, 2)
    assert phi.shape == (2, 2)
    assert theta.shape == (2, 2)
    assert overlap_scores.shape == (2, 2)
    assert np.all(overlap_scores >= 0)


@patch("src.add_ms2lda.main.spectra_to_documents")
def test_run_overlap_scores_calculation_empty_document(
    mock_spectra_to_docs, mock_lda_model, mock_params
) -> None:
    spec1 = Spectrum(mz=np.array([100.1]), intensities=np.array([10.0]))
    mock_spectra_to_docs.return_value = [[]]  # Empty word list

    beta, phi, theta, overlap_scores = run_overlap_scores_calculation(
        [spec1], mock_lda_model, mock_params
    )

    assert overlap_scores.shape == (2, 1)
    assert np.all(theta[:, 0] == 0.0)


# --- Integration & File Handling Tests ---


def test_main_file_not_found(mock_params) -> None:
    mock_params.model = "non_existent_file.bin"
    with pytest.raises(AssertionError, match="Model file does not exist"):
        main(mock_params)


def test_main_model_path_is_directory(mock_params, tmp_path) -> None:
    mock_params.model = str(tmp_path)
    with pytest.raises(AssertionError, match="is a directory, not a file"):
        main(mock_params)


@patch("src.add_ms2lda.main.write_cx")
@patch("src.add_ms2lda.main.read_cx")
@patch("src.add_ms2lda.main.run_overlap_scores_calculation")
@patch("src.add_ms2lda.main.tp.LDAModel.load")
@patch("src.add_ms2lda.main.Path.is_file", return_value=True)
@patch("src.add_ms2lda.main.Path.exists", return_value=True)
def test_main_success(
    mock_exists,
    mock_is_file,
    mock_lda_load,
    mock_calc,
    mock_read_cx,
    mock_write_cx,
    mock_params,
    mock_graph,
    mock_lda_model,
) -> None:
    mock_read_cx.return_value = mock_graph
    mock_lda_load.return_value = mock_lda_model

    # Mock overlap scores (2 topics x 2 nodes)
    # Topic 0 score > threshold (0.2) for node 0, Topic 1 score > threshold for node 1
    mock_overlap = np.array([[0.5, 0.1], [0.0, 0.8]])
    mock_calc.return_value = (None, None, None, mock_overlap)

    main(mock_params)

    # Verify motif annotation on graph nodes
    assert mock_graph.nodes[0]["motifs"] == "0"
    assert mock_graph.nodes[0]["motif_0"] == 1
    assert mock_graph.nodes[0]["motif_1"] == 0

    assert mock_graph.nodes[1]["motifs"] == "1"
    assert mock_graph.nodes[1]["motif_0"] == 0
    assert mock_graph.nodes[1]["motif_1"] == 1

    # Verify write_cx was invoked
    mock_write_cx.assert_called_once()
