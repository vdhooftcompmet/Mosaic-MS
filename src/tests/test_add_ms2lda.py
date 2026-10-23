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
from matchms import Spectrum

from src.add_ms2lda.add_ms2lda import (
    add_ms2lda,
    parse_spectrum_peaks,
)

# --- Fixtures ---


@pytest.fixture
def mock_params():
    """Fixture providing CLI/Config arguments as a Namespace object."""
    return Namespace(
        motifs="dummy_motifs.mgf",
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


# --- Pipeline Integration Test with Mocks ---


@patch("src.add_ms2lda.add_ms2lda.write_cx")
@patch("src.add_ms2lda.add_ms2lda.read_cx")
@patch("src.add_ms2lda.add_ms2lda._calc_overlap_scores")
@patch("src.add_ms2lda.add_ms2lda.load_from_mgf")
def test_add_ms2lda_success(
    mock_load_mgf,
    mock_calc_overlap,
    mock_read_cx,
    mock_write_cx,
    mock_params,
    mock_graph,
) -> None:
    mock_read_cx.return_value = mock_graph
    mock_load_mgf.return_value = [
        Spectrum(np.array([100.1]), np.array([1.0]), {"id": "motif_0"})
    ]

    # Mock overlap scores matching 1 motif x 2 nodes (columns)
    mock_overlap = np.array([[0.5, 0.1]])
    mock_calc_overlap.return_value = mock_overlap

    add_ms2lda(mock_params)

    # Verify motif annotations and attributes assigned to graph nodes
    assert mock_graph.nodes[0]["motifs"] == "0"
    assert mock_graph.nodes[0]["motif_0"] == 0.5  # Changed from 1 to 0.5

    assert mock_graph.nodes[1]["motifs"] == ""
    assert mock_graph.nodes[1]["motif_0"] == 0

    # Verify graph serialization was called
    mock_write_cx.assert_called_once()
