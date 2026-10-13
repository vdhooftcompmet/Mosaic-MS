from argparse import Namespace
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from src.add_snapms.add_snapms import add_snapms_data

# --- Fixtures ---


@pytest.fixture
def mock_config():
    """Fixture providing configuration arguments as a Namespace object."""
    return Namespace(
        graph="dummy_graph.cx",
        snapms="dummy_snapms_folder",
    )


@pytest.fixture
def mock_mn_graph():
    """Fixture providing a NetworkX molecular network graph."""
    G = nx.Graph()
    G.add_node("node_1", mn_cluster_id="cluster_a")
    G.add_node("node_2", mn_cluster_id="cluster_b")
    return G


@pytest.fixture
def mock_annotation_graph():
    """Fixture providing a mocked annotation NetworkX graph."""
    annot_g = nx.Graph()
    annot_g.add_node("annot_1", ann_mass_diversity=5.0, is_top_candidate=True)
    return annot_g


# --- Integration Test with Mocks ---


@patch("src.add_snapms.add_snapms.write_cx")
@patch("src.add_snapms.add_snapms.read_annotations")
@patch("src.add_snapms.add_snapms.read_cx")
def test_add_snapms_data_success(
    mock_read_cx,
    mock_read_annotations,
    mock_write_cx,
    mock_config,
    mock_mn_graph,
    mock_annotation_graph,
):
    mock_read_cx.return_value = mock_mn_graph
    mock_read_annotations.return_value = (
        [mock_annotation_graph],
        ["dummy_annot_file.cx"],
        ["cluster_a"],
    )

    add_snapms_data(mock_config)

    # Verify nodes belonging to annotated clusters receive correct metrics
    assert mock_mn_graph.nodes["node_1"]["ann_mass_diversity"] == 5.0
    assert mock_mn_graph.nodes["node_1"]["is_annotated"] is True

    # Verify nodes belonging to unannotated clusters receive default fallback values
    assert mock_mn_graph.nodes["node_2"]["ann_mass_diversity"] == 0
    assert mock_mn_graph.nodes["node_2"]["is_annotated"] is False

    # Verify write_cx was invoked for both the main graph and the annotation files
    assert mock_write_cx.call_count >= 2
