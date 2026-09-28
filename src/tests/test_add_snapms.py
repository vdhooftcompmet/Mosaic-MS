from argparse import Namespace
from unittest.mock import patch

import networkx as nx
import pytest

from add_snapms.main import add_snapms_data, main

# --- Fixtures ---


@pytest.fixture
def mock_params(tmp_path):
    """Fixture providing CLI arguments as a Namespace object."""
    return Namespace(
        graph=tmp_path / "network.cx2",
        snapms=tmp_path / "snapms_results",
    )


@pytest.fixture
def sample_mass_network():
    """Sample NetworkX graph representing the main mass network."""
    G = nx.Graph()
    G.add_node(0, mn_cluster_id="0")
    G.add_node(1, mn_cluster_id="0")
    G.add_node(2, mn_cluster_id="1")
    G.add_node(3, mn_cluster_id="2")  # Unannotated cluster
    return G


@pytest.fixture
def sample_annotation_graphs():
    """Sample annotation subgraphs corresponding to cluster IDs."""
    # Cluster 0 annotation graph
    ann_0 = nx.Graph()
    ann_0.add_node(10, ann_mass_diversity=3, is_top_candidate=True)
    ann_0.add_node(11, ann_mass_diversity=1, is_top_candidate=False)

    # Cluster 1 annotation graph
    ann_1 = nx.Graph()
    ann_1.add_node(20, ann_mass_diversity=2, is_top_candidate=False)

    return {"0": ann_0, "1": ann_1}


# --- Unit Tests for add_snapms_data ---


def test_add_snapms_data_annotations(sample_mass_network, sample_annotation_graphs) -> None:
    add_snapms_data(sample_mass_network, sample_annotation_graphs)

    # Cluster 0 nodes should receive max_diversity = 3 and is_annotated = True
    assert sample_mass_network.nodes[0]["ann_mass_diversity"] == 3
    assert sample_mass_network.nodes[0]["is_annotated"] is True
    assert sample_mass_network.nodes[1]["ann_mass_diversity"] == 3
    assert sample_mass_network.nodes[1]["is_annotated"] is True

    # Cluster 1 nodes should receive max_diversity = 2 and is_annotated = False (no top candidate)
    assert sample_mass_network.nodes[2]["ann_mass_diversity"] == 2
    assert sample_mass_network.nodes[2]["is_annotated"] is False


def test_add_snapms_data_unannotated_cluster(
    sample_mass_network, sample_annotation_graphs
) -> None:
    add_snapms_data(sample_mass_network, sample_annotation_graphs)

    # Cluster 2 (not in annotations dict) should default to 0 diversity and False for is_annotated
    assert sample_mass_network.nodes[3]["ann_mass_diversity"] == 0
    assert sample_mass_network.nodes[3]["is_annotated"] is False


def test_add_snapms_data_empty_annotation_graph(sample_mass_network) -> None:
    empty_ann = nx.Graph()  # Graph with no nodes
    annotations = {"0": empty_ann}

    add_snapms_data(sample_mass_network, annotations)

    # Nodes in Cluster 0 should handle empty graphs gracefully
    assert sample_mass_network.nodes[0]["ann_mass_diversity"] == 0
    assert sample_mass_network.nodes[0]["is_annotated"] is False


# --- Integration Test for main ---


@patch("add_snapms.main.write_cx")
@patch("add_snapms.main.read_annotations")
@patch("add_snapms.main.read_cx")
def test_main_execution(
    mock_read_cx,
    mock_read_annotations,
    mock_write_cx,
    mock_params,
    sample_mass_network,
    sample_annotation_graphs,
) -> None:
    mock_read_cx.return_value = sample_mass_network

    annotations = list(sample_annotation_graphs.values())
    files = ["file_0.cx2", "file_1.cx2"]
    cluster_ids = [0, 1]
    mock_read_annotations.return_value = (annotations, files, cluster_ids)

    main(mock_params)

    # Check input functions were called with parameters
    mock_read_cx.assert_called_once_with(str(mock_params.graph))
    mock_read_annotations.assert_called_once_with(mock_params.snapms)

    # Check that write_cx was called once for the main graph + once for each annotation file
    assert mock_write_cx.call_count == 1 + len(files)

    # Verify nodes in the main graph were modified properly during main execution
    assert sample_mass_network.nodes[0]["ann_mass_diversity"] == 3
    assert sample_mass_network.nodes[0]["is_annotated"] is True
