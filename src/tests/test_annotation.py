import json
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="builtin type _VocabDict has no __module__ attribute",
)

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
from matchms import Spectrum

from src.annotation.annotation import (
    FragmentMetadata,
    _add_cluster_numbering,
    _deduplicate_fragments,
    _derive_motif,
    _normalize_intensities,
    build_spectra_graph,
    fragment_snapms,
)
from src.utils.configs import AnnotationConfig


@pytest.fixture
def mock_annotation_config():
    """Provides a dummy AnnotationConfig object."""
    config = MagicMock(spec=AnnotationConfig)
    config.graph = "dummy_graph.cx2"
    config.result_folder = "dummy_results"
    config.min_cluster_size = 1
    config.max_cluster_size = 10
    config.cutoff = 0.5
    return config


# --- Unit Tests for Helpers ---


def test_normalize_intensities() -> None:
    arr = np.array([2.0, 4.0, 1.0])
    normalized = _normalize_intensities(arr)
    np.testing.assert_array_almost_equal(normalized, [0.5, 1.0, 0.25])

    empty = np.array([])
    np.testing.assert_array_equal(_normalize_intensities(empty), empty)

    zeros = np.array([0.0, 0.0])
    np.testing.assert_array_equal(_normalize_intensities(zeros), zeros)


def test_deduplicate_fragments() -> None:
    f1 = FragmentMetadata(
        mz=100.0,
        variation="[M]",
        mass_error_da=0.01,
        smiles="CC",
        morgan_fingerprint="base64fp",
        mass_cluster={100},
    )
    f2 = FragmentMetadata(
        mz=100.0,
        variation="[M]",
        mass_error_da=0.01,
        smiles="CC",
        morgan_fingerprint="base64fp",
        mass_cluster={200},
    )
    deduped = _deduplicate_fragments([f1, f2])
    assert len(deduped) == 1
    assert deduped[0].mass_cluster == {100, 200}


def test_add_cluster_numbering() -> None:
    G = nx.Graph()
    G.add_node(0, mass_cluster={100, 200})
    G.add_node(1, mass_cluster={100, 200})
    G.add_edge(0, 1)  # Connect nodes so they form a single connected component
    _add_cluster_numbering(G)
    assert "parent_cluster" in G.nodes[0]
    assert G.nodes[0]["parent_cluster"] == G.nodes[1]["parent_cluster"]


def test_derive_motif() -> None:
    G = nx.Graph()
    G.add_node(
        "node_1",
        motif_peaks_json=json.dumps([[100.0, 10.0], [200.0, 5.0]]),
        motif_charge=1,
    )
    spectrum = _derive_motif(G, "node_1")
    assert isinstance(spectrum, Spectrum)
    assert len(spectrum.peaks.mz) == 2
    assert spectrum.get("charge") == 1  # Expect integer 1


def test_derive_motif_invalid_peaks_raises() -> None:
    G = nx.Graph()
    G.add_node("node_1", motif_charge=1)
    with pytest.raises(ValueError):
        _derive_motif(G, "node_1")


@patch("src.annotation.annotation._detect_motifs")
def test_build_spectra_graph(mock_detect_motifs) -> None:
    mock_detect_motifs.return_value = [[0]]
    motif = Spectrum(
        mz=np.array([100.0], dtype=float),
        intensities=np.array([1.0], dtype=float),
        metadata={"id": "motif_0"},
    )
    spectrum = Spectrum(
        mz=np.array([100.0], dtype=float),
        intensities=np.array([1.0], dtype=float),
        metadata={"id": "spec_0"},
    )

    G = build_spectra_graph([motif], [spectrum])
    assert isinstance(G, nx.Graph)
    assert len(G.nodes) == 1
    node_key = list(G.nodes)[0]
    assert G.nodes[node_key]["mn_cluster_id"] == 0


# --- Pipeline Integration Test ---


@patch("src.annotation.annotation.write_cx")
@patch("src.annotation.annotation.prepare_directory")
@patch("pathlib.Path.is_dir", return_value=False)
@patch("pathlib.Path.exists", return_value=False)
@patch("src.annotation.annotation.group_nodes")
@patch("src.annotation.annotation.import_atlas")
@patch("src.annotation.annotation.read_cx")
def test_fragment_snapms(
    mock_read_cx,
    mock_import_atlas,
    mock_group_nodes,
    mock_exists,
    mock_is_dir,
    mock_prepare_dir,
    mock_write_cx,
    mock_annotation_config,
) -> None:
    mock_graph = nx.Graph()
    mock_graph.add_node(0, peaks_json="[[100.0, 1.0]]")
    mock_read_cx.return_value = mock_graph
    mock_group_nodes.return_value = {"cluster_1": {0}}

    with patch("src.annotation.annotation._find_fragment_matches", return_value=[]):
        fragment_snapms(mock_annotation_config, graph=mock_graph)

    mock_group_nodes.assert_called_once()
