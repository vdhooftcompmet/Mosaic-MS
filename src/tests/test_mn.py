from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
from matchms import Spectrum

from mn.mn import (
    EdgeData,
    _assign_cluster_ids,
    _extract_graphdata,
    _filter_components,
    _mask_spectra_globally,
    add_cluster_numbering,
    bin_spectra,
    build_graph,
    calculate_bootstrapping,
    filter_base_strategy,
    filter_rescue_strategy,
    filter_threshold_strategy,
    get_similarity,
    global_bins,
    mutual_topk,
    plain_similarity,
    run_bootstrap,
    run_networking,
)
from utils.configs import MNConfig

# --- Fixtures ---


@pytest.fixture
def mock_config():
    """Mock config with preset defaults for MN testing."""
    config = MagicMock(spec=MNConfig)
    config.similarity_type = "cosine"
    config.flash_tolerance = 0.1
    config.binning_decimals = 2
    config.seed = 42
    config.B = 5
    config.k = 2
    config.similarity_threshold = 0.7
    config.support_threshold = 0.3
    config.rescue_similarity_threshold = 0.2
    config.max_component_size = 5
    config.ms2deepscore_model_path = "dummy_ms2ds.pt"
    config.spec2vec_model_path = "dummy_s2v.model"
    return config


@pytest.fixture
def sample_spectra():
    """Sample spectra list for matchms operations."""
    spec1 = Spectrum(
        mz=np.array([100.123, 200.456, 300.789], dtype=float),
        intensities=np.array([10.0, 50.0, 100.0], dtype=float),
        metadata={"compound_name": "Compound A", "id": "A"},
    )
    spec2 = Spectrum(
        mz=np.array([100.124, 200.457, 400.111], dtype=float),
        intensities=np.array([15.0, 45.0, 80.0], dtype=float),
        metadata={"compound_name": "Compound B", "id": "B"},
    )
    spec3 = Spectrum(
        mz=np.array([150.000, 250.000, 350.000], dtype=float),
        intensities=np.array([5.0, 25.0, 75.0], dtype=float),
        metadata={"compound_name": "Compound C", "id": "C"},
    )
    return [spec1, spec2, spec3]


# --- Spectrum Binning & Masking Tests ---


def test_global_bins(sample_spectra):
    bins = global_bins(sample_spectra, decimals=2)

    assert isinstance(bins, np.ndarray)
    assert len(bins) == len(set(bins))  # All unique
    assert np.array_equal(bins, np.sort(bins))  # Sorted
    assert 100.12 in bins
    assert 200.46 in bins


def test_bin_spectra(sample_spectra):
    binned = bin_spectra(sample_spectra, decimals=2)

    assert len(binned) == len(sample_spectra)
    assert np.allclose(binned[0].peaks.mz, np.array([100.12, 200.46, 300.79]))
    assert binned[0].metadata["compound_name"] == "Compound A"


def test_mask_spectra_globally(sample_spectra):
    bins = global_bins(sample_spectra, decimals=2)
    binned = bin_spectra(sample_spectra, decimals=2)
    rng = np.random.default_rng(42)

    masked = _mask_spectra_globally(rng, bins, binned)

    assert len(masked) == len(sample_spectra)
    for spec in masked:
        assert isinstance(spec, Spectrum)
        assert len(spec.peaks.mz) >= 1
        assert len(spec.peaks.intensities) == len(spec.peaks.mz)


# --- Matrix Operations Tests ---


def test_mutual_topk():
    A = np.array(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.8, 0.3],
            [0.1, 0.8, 1.0, 0.85],
            [0.2, 0.3, 0.85, 1.0],
        ]
    )
    k = 1

    result = mutual_topk(A, k)

    # (0, 1) mutually top-1 each other
    assert result[0, 1] == 0.9
    assert result[1, 0] == 0.9
    # (2, 3) mutually top-1 each other
    assert result[2, 3] == 0.85
    assert result[3, 2] == 0.85
    # Non-mutual top-k zeroed out
    assert result[0, 2] == 0.0


# --- Model / Similarity Factory Tests ---


@pytest.mark.parametrize("method_name", ["cos", "cosine", "modcos", "modified_cosine"])
def test_get_similarity_flash(method_name, mock_config):
    metric = get_similarity(method_name, 0.1, mock_config)
    assert metric is not None


def test_get_similarity_ms2ds_missing_file(mock_config):
    mock_config.ms2deepscore_model_path = "non_existent_file.pt"
    with pytest.raises(FileNotFoundError, match="not found"):
        get_similarity("ms2ds", 0.1, mock_config)


@patch("mn.mn.Path.exists", return_value=True)
@patch("mn.mn.load_model")
def test_get_similarity_ms2ds_success(mock_load_model, mock_exists, mock_config):
    mock_load_model.return_value = MagicMock()
    metric = get_similarity("ms2ds", 0.1, mock_config)
    assert metric is not None


def test_get_similarity_spec2vec_missing_file(mock_config):
    mock_config.spec2vec_model_path = "non_existent_file.model"
    with pytest.raises(FileNotFoundError, match="not found"):
        get_similarity("s2v", 0.1, mock_config)


@patch("mn.mn.Path.exists", return_value=True)
@patch("mn.mn.gensim.models.Word2Vec.load")
def test_get_similarity_spec2vec_success(mock_w2v_load, mock_exists, mock_config):
    mock_model = MagicMock()
    mock_model.wv.key_to_index = {"peak@100.12": 0}
    mock_w2v_load.return_value = mock_model

    metric = get_similarity("s2v", 0.1, mock_config)
    assert metric is not None


def test_get_similarity_invalid_option(mock_config):
    with pytest.raises(ValueError, match="unknown option"):
        get_similarity("unknown_method", 0.1, mock_config)


# --- Plain Similarity & Bootstrapping Tests ---


def test_plain_similarity(sample_spectra, mock_config):
    sim_matrix = plain_similarity(sample_spectra, mock_config)

    assert isinstance(sim_matrix, np.ndarray)
    assert sim_matrix.shape == (3, 3)
    assert np.allclose(np.diag(sim_matrix), 1.0)


def test_calculate_bootstrapping(sample_spectra, mock_config):
    mean_sim, mean_sup = calculate_bootstrapping(sample_spectra, mock_config)

    assert mean_sim.shape == (3, 3)
    assert mean_sup.shape == (3, 3)
    assert np.allclose(np.diag(mean_sim), 1.0)
    assert np.all((mean_sup >= 0.0) & (mean_sup <= 1.0))


@patch("mn.mn.calculate_bootstrapping")
def test_run_bootstrap(mock_calc, sample_spectra, mock_config):
    mock_calc.return_value = (np.ones((3, 3)), np.zeros((3, 3)))
    sim, sup = run_bootstrap(sample_spectra, mock_config)

    assert sim.shape == (3, 3)
    assert sup.shape == (3, 3)
    mock_calc.assert_called_once()


# --- Edge Filtering Strategies Tests ---


def test_filter_base_strategy():
    strategy = filter_base_strategy(sim_threshold=0.7)
    edge_data = EdgeData(
        u=np.array([0, 0, 1]),
        v=np.array([1, 2, 2]),
        sim=np.array([0.8, 0.5, 0.7]),
        sup=np.array([0.5, 0.5, 0.5]),
        lbl=np.array(["", "", ""]),
    )

    filtered = strategy(edge_data)
    assert list(filtered.u) == [0, 1]
    assert list(filtered.v) == [1, 2]


def test_filter_threshold_strategy():
    strategy = filter_threshold_strategy(sim_threshold=0.7, support_threshold=0.5)
    edge_data = EdgeData(
        u=np.array([0, 0, 1]),
        v=np.array([1, 2, 2]),
        sim=np.array([0.8, 0.9, 0.7]),
        sup=np.array([0.6, 0.2, 0.5]),
        lbl=np.array(["", "", ""]),
    )

    filtered = strategy(edge_data)
    assert list(filtered.u) == [0, 1]
    assert list(filtered.v) == [1, 2]


def test_filter_rescue_strategy():
    strategy = filter_rescue_strategy(
        sim_core=0.7, support_core=0.3, sim_rescue_min=0.4, support_rescue=0.5
    )
    edge_data = EdgeData(
        u=np.array([0, 0, 1]),
        v=np.array([1, 2, 2]),
        sim=np.array([0.8, 0.5, 0.3]),
        sup=np.array([0.4, 0.6, 0.9]),
        lbl=np.array(["", "", ""]),
    )

    filtered = strategy(edge_data)
    assert list(filtered.u) == [0, 0]
    assert list(filtered.v) == [1, 2]
    assert list(filtered.lbl) == ["core", "rescued"]


# --- Component & Graph Building Tests ---


def test_extract_graphdata(sample_spectra):
    sim = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.9], [0.2, 0.9, 1.0]])
    sup = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.7], [0.1, 0.7, 1.0]])

    G, edge_data = _extract_graphdata(sample_spectra, sim, sup)

    assert G.number_of_nodes() == 3
    assert len(edge_data.u) == 3  # Upper triangle pairs (0,1), (0,2), (1,2)
    assert list(edge_data.sim) == [0.8, 0.2, 0.9]


def test_filter_components_max_size():
    edge_data = EdgeData(
        u=np.array([0, 1, 2]),
        v=np.array([1, 2, 3]),
        sim=np.array([0.9, 0.8, 0.7]),
        sup=np.array([1.0, 1.0, 1.0]),
        lbl=np.array(["", "", ""]),
    )

    # Edge (0,1) merges size 1+1=2
    # Edge (1,2) merges size 2+1=3
    # Edge (2,3) would merge size 3+1=4 > max_component_size (3), so both components retire
    filtered = _filter_components(edge_data, max_component_size=3, retire_groups=True)

    assert len(filtered.u) == 2
    assert list(filtered.u) == [0, 1]


def test_build_graph(sample_spectra):
    sim = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.9], [0.1, 0.9, 1.0]])
    sup = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.7], [0.1, 0.7, 1.0]])
    strategy = filter_base_strategy(sim_threshold=0.5)

    G = build_graph(
        sample_spectra, sim, sup, filter_strategy=strategy, max_component_size=5
    )

    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert "component" in G.nodes[0]
    assert G[0][1]["weight"] == 0.8


def test_run_networking(sample_spectra, mock_config):
    sim = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.9], [0.1, 0.9, 1.0]])
    sup = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.7], [0.1, 0.7, 1.0]])

    G = run_networking(
        sample_spectra, sim, sup, network_type="threshold", config=mock_config
    )

    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3


def test_assign_and_add_cluster_numbering():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (3, 4)])

    _assign_cluster_ids(G)
    add_cluster_numbering(G)

    # Connected component of size 3 should be component 0, size 2 component 1
    assert G.nodes[0]["component"] == 0
    assert G.nodes[3]["component"] == 1

    assert G.nodes[0]["mn_cluster_id"] == 0
    assert G.nodes[3]["mn_cluster_id"] == 1
