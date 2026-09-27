from unittest.mock import MagicMock

import networkx as nx
import pandas as pd
import pytest
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from snapms.snapms import (
    _nr_of_unique_compounds,
    add_cluster_numbering,
    add_top_candidate_annotation,
    compute_adduct_matches,
    derive_neutral_mass,
    filter_clusters,
    get_adducts,
    get_edges,
    get_unique_id,
    group_by_property,
    import_atlas,
    merge_duplicates,
    remove_edges_with_same_value_for,
    remove_self_similar_vals,
    remove_small_subgraphs,
)
from utils.configs import SNAPMSConfig

# --- Fixtures ---


@pytest.fixture
def mock_config():
    """Mock config with preset defaults for SNAPMS testing."""
    config = MagicMock(spec=SNAPMSConfig)
    config.reference_db = "dummy_path.json"
    config.detect_adduct = True
    config.adduct_list = ["[M+H]+"]
    config.ppm_error = 10
    config.min_cluster_size = 2
    config.max_cluster_size = 5
    config.min_annotation_size = 2
    return config


@pytest.fixture
def sample_db_df():
    """Mock DataFrame mimicking reference database output."""
    fp1 = ExplicitBitVect(2048)
    fp1.SetBitsFromList([1, 10, 100])

    fp2 = ExplicitBitVect(2048)
    fp2.SetBitsFromList([1, 10, 200])

    return pd.DataFrame(
        [
            {
                "neutral_mass": 300.0,
                "smiles": "CCO",
                "inchikey": "INCHIKEY1",
                "morgan_fingerprint": fp1.ToBase64(),
            },
            {
                "neutral_mass": 350.0,
                "smiles": "CCN",
                "inchikey": "INCHIKEY2",
                "morgan_fingerprint": fp2.ToBase64(),
            },
        ]
    )


@pytest.fixture
def sample_network():
    """Sample NetworkX graph representing mass networks."""
    G = nx.Graph()
    G.add_node("node_1", precursor_mz=301.008, adduct="[M+H]+", motifs="1;2", cluster=1)
    G.add_node(
        "node_2", precursor_mz=322.989218, adduct="[M+Na]+", motifs="2;3", cluster=1
    )
    return G


# --- Database Import Tests ---


def test_import_atlas_file_not_found(mock_config, tmp_path):
    mock_config.reference_db = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError, match="Reference DB file not found"):
        import_atlas(mock_config)


def test_import_atlas_success(mock_config, tmp_path):
    db_file = tmp_path / "db.json"
    db_file.write_text('{"neutral_mass": 100.0, "smiles": "C"}\n')
    mock_config.reference_db = db_file

    df = import_atlas(mock_config)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["smiles"] == "C"


# --- Mass & Adduct Tests ---


@pytest.mark.parametrize(
    "precursor_mz, adduct, expected_neutral_mass",
    [
        (101.0080, "[M+H]+", 100.0),
        (101.0080, "m_plus_h", 100.0),
        (122.989218, "[M+Na]+", 100.0),
        (138.963158, "[M+K]+", 100.0),
        (201.0080, "[2M+H]+", 100.0),
        (100.0000, "[M]+", 100.0),
    ],
)
def test_derive_neutral_mass_valid(precursor_mz, adduct, expected_neutral_mass):
    calculated = derive_neutral_mass(precursor_mz, adduct)
    pytest.approx(calculated, expected_neutral_mass, abs=1e-4)


def test_derive_neutral_mass_invalid():
    with pytest.raises(ValueError, match="Adduct not recognized"):
        derive_neutral_mass(100.0, "[UNKNOWN]+")


def test_get_adducts_detect_adduct_false(sample_network, mock_config):
    mock_config.detect_adduct = False
    mock_config.adduct_list = ["[M+H]+", "[M+Na]+"]
    adducts = get_adducts(sample_network, "node_1", mock_config)
    assert adducts == ["[M+H]+", "[M+Na]+"]


def test_get_adducts_detect_adduct_true_found(sample_network, mock_config):
    mock_config.detect_adduct = True
    mock_config.adduct_list = ["[M+H]+"]
    adducts = get_adducts(sample_network, "node_1", mock_config)
    assert set(adducts) == {"[M+H]+"}


def test_get_adducts_fallback_warning(sample_network, mock_config, capsys):
    mock_config.detect_adduct = True
    mock_config.adduct_list = ["[M+H]+"]
    sample_network.add_node("node_3", precursor_mz=100.0)

    adducts = get_adducts(sample_network, "node_3", mock_config)
    captured = capsys.readouterr()

    assert adducts == ["[M+H]+"]
    assert "WARNING: no adduct found" in captured.out


# --- Database Matching Tests ---


def test_compute_adduct_matches(sample_network, sample_db_df, mock_config):
    matches = compute_adduct_matches(
        sample_network, ["node_1"], mock_config, sample_db_df
    )

    assert len(matches) == 1
    assert matches[0]["mn_node_id"] == "node_1"
    assert matches[0]["smiles"] == "CCO"
    assert matches[0]["adduct"] == "[M+H]+"


def test_compute_adduct_matches_missing_mass_or_invalid_adduct(
    sample_network, sample_db_df, mock_config, capsys
):
    sample_network.add_node("node_no_mass")
    sample_network.add_node("node_bad_adduct", precursor_mz=100.0, adduct="BAD")

    matches = compute_adduct_matches(
        sample_network, ["node_no_mass", "node_bad_adduct"], mock_config, sample_db_df
    )
    captured = capsys.readouterr()

    assert len(matches) == 0
    assert "precursor mass not found" in captured.out
    assert "unknown adduct" in captured.out


def test_merge_duplicates():
    matches = [
        {"smiles": "CCO", "mn_node_id": "node_1", "motifs": "1;2"},
        {"smiles": "CCO", "mn_node_id": "node_2", "motifs": "2;3"},
    ]

    merged = merge_duplicates(matches)

    assert len(merged) == 1
    m = merged[0]
    assert set(m["mn_node_id"].split(";")) == {"node_1", "node_2"}
    assert set(m["motifs"].split(";")) == {"1", "2", "3"}
    assert m["parent_node_node_1"] == 1
    assert m["parent_node_node_2"] == 1


# --- Clustering & Edge Calculations ---


def test_group_by_property(sample_network):
    groups = group_by_property(sample_network, "cluster")
    assert 1 in groups
    assert len(groups[1]) == 2


def test_filter_clusters(mock_config):
    clusters = {
        "c1": {1: {}},  # Size 1 (too small)
        "c2": {1: {}, 2: {}, 3: {}},  # Size 3 (valid)
        "c3": {i: {} for i in range(10)},  # Size 10 (too large)
    }
    filtered = filter_clusters(clusters, mock_config)
    assert set(filtered.keys()) == {"c2"}


def test_get_edges(sample_db_df):
    matches = sample_db_df.to_dict(orient="records")
    # Tweak neutral_masses so they differ and can form valid edges
    matches[0]["neutral_mass"] = 300.0
    matches[1]["neutral_mass"] = 350.0

    edges = get_edges(matches, cutoff=0.1)
    assert edges == [(0, 1)]


def test_get_edges_ignores_equal_neutral_mass(sample_db_df):
    matches = sample_db_df.to_dict(orient="records")
    matches[0]["neutral_mass"] = 300.0
    matches[1]["neutral_mass"] = 300.0  # Same neutral mass

    edges = get_edges(matches, cutoff=0.1)
    assert len(edges) == 0


def test_remove_self_similar_vals():
    edges = [(0, 1), (1, 1), (2, 3)]
    filtered = remove_self_similar_vals(edges)
    assert filtered == [(0, 1), (2, 3)]


def test_remove_edges_with_same_value_for():
    edges = [(0, 1), (1, 2)]
    metadata = {0: {"type": "A"}, 1: {"type": "A"}, 2: {"type": "B"}}

    filtered = remove_edges_with_same_value_for(edges, metadata, "type")
    assert filtered == [(1, 2)]


# --- Graph Candidate & Clustering Annotations ---


def test_remove_small_subgraphs(mock_config):
    G = nx.Graph()
    G.add_edges_from([(1, 2)])  # Cluster size 2
    G.add_node(3)  # Cluster size 1

    remove_small_subgraphs(G, mock_config)

    assert set(G.nodes()) == {1, 2}


def test_add_top_candidate_annotation():
    G = nx.Graph()
    # Cluster 1: 3 unique mn_node_id compounds
    G.add_node(1, mn_node_id="A")
    G.add_node(2, mn_node_id="B")
    G.add_node(3, mn_node_id="C")
    G.add_edges_from([(1, 2), (2, 3)])

    # Cluster 2: 1 unique mn_node_id compound
    G.add_node(4, mn_node_id="D")

    add_top_candidate_annotation(G)

    assert G.nodes[1]["is_top_candidate"] is True
    assert G.nodes[1]["ann_mass_diversity"] == 3
    assert G.nodes[4]["is_top_candidate"] is False
    assert G.nodes[4]["ann_mass_diversity"] == 1


def test_add_top_candidate_annotation_max_count_le_2():
    G = nx.Graph()
    G.add_node(1, mn_node_id="A")
    G.add_node(2, mn_node_id="B")
    G.add_edge(1, 2)

    add_top_candidate_annotation(G)

    assert G.nodes[1]["is_top_candidate"] is False
    assert G.nodes[2]["is_top_candidate"] is False


def test_nr_of_unique_compounds():
    G = nx.Graph()
    G.add_node(1, key="A")
    G.add_node(2, key="A")
    G.add_node(3, key="B")

    count = _nr_of_unique_compounds(G, {1, 2, 3}, "key")
    assert count == 2


def test_get_unique_id_and_add_cluster_numbering():
    id1 = get_unique_id()
    id2 = get_unique_id()
    assert id2 == id1 + 1

    G = nx.Graph()
    G.add_edges_from([(1, 2), (3, 4)])

    add_cluster_numbering(G)

    cluster_ids = {G.nodes[node]["mn_cluster_id"] for node in G.nodes()}
    assert len(cluster_ids) == 2
