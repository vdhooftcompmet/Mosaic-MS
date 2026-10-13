import json
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="builtin type _VocabDict has no __module__ attribute",
)
from argparse import Namespace
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from src.add_ms2lda.add_ms2lda import (
    add_ms2lda,
    parse_spectrum_peaks,
)

# --- Fixtures ---


@pytest.fixture
def mock_params():
    """Fixture providing CLI/Config arguments as a Namespace object."""
    return Namespace(
        model="dummy_model.bin",
        graph="dummy_graph.cx2",
        threshold=0.2,
    )


@pytest.fixture
def mock_graph():
    """Fixture providing a NetworkX graph populated with peak JSON metadata and list formats."""
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
    return model


# --- Unit Tests for Helpers ---


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


# --- Validation Tests ---


def test_add_ms2lda_file_not_found(mock_params) -> None:
    mock_params.model = "non_existent_file.bin"
    with pytest.raises(AssertionError, match="Model file does not exist"):
        add_ms2lda(mock_params)


def test_add_ms2lda_model_path_is_directory(mock_params, tmp_path) -> None:
    mock_params.model = str(tmp_path)
    with pytest.raises(AssertionError, match="is a directory, not a file"):
        add_ms2lda(mock_params)


# --- Pipeline Integration Test with Mocks ---


@patch("src.add_ms2lda.add_ms2lda.write_cx")
@patch("src.add_ms2lda.add_ms2lda.read_cx")
@patch("src.utils.motifs.run_overlap_scores_calculation")
@patch("src.add_ms2lda.add_ms2lda.tp.LDAModel.load")
@patch("src.add_ms2lda.add_ms2lda.Path.is_file", return_value=True)
@patch("src.add_ms2lda.add_ms2lda.Path.exists", return_value=True)
def test_add_ms2lda_success(
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

    # Mock overlap scores matching 2 topics x 2 nodes (columns)
    mock_overlap = np.array([[0.5, 0.1], [0.0, 0.8]])
    mock_calc.return_value = (None, None, None, mock_overlap)

    add_ms2lda(mock_params)

    # Verify motif annotations and attributes assigned to graph nodes
    assert mock_graph.nodes[0]["motifs"] == "0"
    assert mock_graph.nodes[0]["motif_0"] == 1
    assert mock_graph.nodes[0]["motif_1"] == 0

    assert mock_graph.nodes[1]["motifs"] == "1"
    assert mock_graph.nodes[1]["motif_0"] == 0
    assert mock_graph.nodes[1]["motif_1"] == 1

    # Verify graph serialization was called
    mock_write_cx.assert_called_once()
