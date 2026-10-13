import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
from matchms import Spectrum
from rdkit import Chem

from src.annotation.annotation import (
    clear_folder,
    construct_motif_graph,
    derive_motif,
    derive_valid_fragments,
    filter_valid_mass,
    find_fragment_matches,
    fragment_snapms,
    normalize_intensities,
    topic_to_motif,
)
from src.snapms.snapms import Adduct, DBMatch

# --- Fixtures ---


@pytest.fixture
def mock_annotation_config(tmp_path):
    """Fixture providing an AnnotationConfig-like Namespace for testing."""
    result_folder = tmp_path / "results"
    return Namespace(
        model="dummy_model.bin",
        mgf="dummy.mgf",
        graph="dummy_graph.cx",
        threshold=0.2,
        result_folder=str(result_folder),
        min_cluster_size=1,
        max_cluster_size=10,
    )


@pytest.fixture
def mock_lda_model():
    """Fixture providing a mocked tomotopy LDAModel."""
    model = MagicMock()
    model.get_topic_words.side_effect = lambda topic_id, top_n=50: [
        ("frag@100.12", 0.6),
        ("loss@200.45", 0.4),
    ]
    return model


@pytest.fixture
def sample_graph():
    """Fixture providing a NetworkX graph with motif peaks metadata."""
    G = nx.Graph()
    G.add_node(
        "node_1",
        motif_peaks_json=json.dumps([[100.1, 10.0], [200.2, 50.0]]),
        motif_id=0,
        compound_name="TestCompound",
    )
    return G


# --- Unit Tests for Motif and Spectrum Helpers ---


def test_normalize_intensities():
    intensities = np.array([0.0, 5.0, 10.0])
    normalized = normalize_intensities(intensities)
    np.testing.assert_array_equal(normalized, np.array([0.0, 0.5, 1.0]))


def test_normalize_intensities_zeros():
    intensities = np.array([0.0, 0.0])
    normalized = normalize_intensities(intensities)
    np.testing.assert_array_equal(normalized, intensities)


def test_topic_to_motif(mock_lda_model):
    motif = topic_to_motif(mock_lda_model, topic_index=0)

    assert isinstance(motif, Spectrum)
    assert len(motif.peaks.mz) == 2
    np.testing.assert_array_equal(motif.peaks.mz, np.array([100.12, 200.45]))
    np.testing.assert_array_equal(motif.peaks.intensities, np.array([0.6, -0.4]))
    assert motif.metadata["motif_id"] == 0


def test_derive_motif(sample_graph):
    spectrum = derive_motif(sample_graph, "node_1")

    assert isinstance(spectrum, Spectrum)
    assert len(spectrum.peaks.mz) == 2
    np.testing.assert_array_equal(spectrum.peaks.intensities, np.array([0.2, 1.0]))
    assert spectrum.metadata["id"] == 0


def test_derive_motif_empty_peaks():
    G = nx.Graph()
    G.add_node("empty_node", motif_peaks_json="[]")
    spectrum = derive_motif(G, "empty_node")

    assert len(spectrum.peaks.mz) == 0
    assert len(spectrum.peaks.intensities) == 0


# --- Folder Cleanup Utility Test ---


def test_clear_folder(tmp_path):
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    file_path = sub_dir / "test.txt"
    file_path.write_text("some content")

    clear_folder(sub_dir)

    assert file_path.exists()
    assert file_path.read_text() == ""


# --- Mass Filtering & Fragment Derivation Tests ---


def test_filter_valid_mass():
    # Provide multiplier, mass_shift, and charge
    adduct = Adduct(multiplier=1, mass_shift=1.0078, charge=1)
    fragment_mass = 99.09
    valid_masses = np.array([100.1])

    results = list(filter_valid_mass(fragment_mass, adduct, valid_masses))
    assert len(results) > 0
    mz, variation, mass_error = results[0]
    assert mz == 100.1
    assert abs(mass_error) <= 0.1


@patch("src.annotation.annotation.greedy_substructure_finder")
def test_derive_valid_fragments(mock_substructure_finder):
    motif = Spectrum(
        mz=np.array([100.1, 200.2]), intensities=np.array([0.8, 0.2]), metadata={}
    )

    mol = Chem.MolFromSmiles("C")
    mock_substructure_finder.return_value = [mol]

    db_match = DBMatch(
        neutral_mass=16.04,
        smiles="C",
        inchikey="VNWKTOKETHGBQD-UHFFFAOYSA-N",
        morgan_fingerprint="",
        mn_node_id="node_1",
        adduct="[M+H]+",
    )

    with patch(
        "src.annotation.annotation.ADDUCT_RULES",
        {"[M+H]+": Adduct(multiplier=1, mass_shift=1.0, charge=1)},
    ):
        with patch(
            "src.annotation.annotation.filter_valid_mass",
            return_value=[(100.1, "[M+H]", 0.05)],
        ):
            generator = derive_valid_fragments(motif, [db_match], break_rings=False)
            results = list(generator)

            assert len(results) == 1
            substructure, smiles, mz, variation, mass_error = results[0]
            assert smiles == "C"
            assert mz == 100.1
            assert variation == "[M+H]"


# --- Complex Pipeline Mock Tests ---


@patch("src.annotation.annotation.clean_mgf")
@patch("src.annotation.annotation.tp.LDAModel.load")
@patch("src.annotation.annotation.run_overlap_scores_calculation")
def test_construct_motif_graph(
    mock_overlap_calc, mock_lda_load, mock_clean_mgf, mock_annotation_config
):
    mock_spectrum = MagicMock()
    mock_spectrum.to_dict.return_value = {"compound_name": "MockSpec"}
    mock_clean_mgf.return_value = [mock_spectrum]

    overlap_scores = np.array([[0.8]])
    mock_overlap_calc.return_value = (None, None, None, overlap_scores)

    mock_motif = MagicMock()
    mock_motif.to_dict.return_value = {"id": 0}

    with patch("src.annotation.annotation.topic_to_motif", return_value=mock_motif):
        G = construct_motif_graph(mock_annotation_config)

    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() > 0
    assert "m0s0" in G.nodes
    assert G.nodes["m0s0"]["mn_cluster_id"] == 0


@patch("src.annotation.annotation.find_db_matches")
@patch("src.annotation.annotation.derive_motif")
@patch("src.annotation.annotation.derive_valid_fragments")
@patch("src.annotation.annotation.assign_mass_cluster")
@patch("src.annotation.annotation.merge_duplicates", side_effect=lambda x: x)
def test_find_fragment_matches(
    mock_merge, mock_assign, mock_derive_fragments, mock_derive_motif, mock_find_db
):
    mn = nx.Graph()
    mn.add_node("node_1", motif_peaks_json="[]")

    db_match = DBMatch(
        neutral_mass=16.04,
        smiles="C",
        inchikey="VNWKTOKETHGBQD-UHFFFAOYSA-N",
        morgan_fingerprint="",
        mn_node_id="node_1",
        adduct="[M+H]+",
    )
    mock_find_db.return_value = [db_match]

    mock_derive_fragments.return_value = [
        (Chem.MolFromSmiles("C"), "C", 100.1, "[M+H]", 0.01)
    ]
    mock_assign.side_effect = lambda x: x

    atlas_con = MagicMock()
    config = Namespace(min_cluster_size=1, max_cluster_size=10)

    fragments = find_fragment_matches(
        mn, {"node_1"}, config, atlas_con, break_rings=False
    )

    assert isinstance(fragments, list)
    assert len(fragments) == 1
    assert fragments[0].smiles == "C"
    assert fragments[0].mz == 100.1


@patch("src.annotation.annotation.read_cx")
@patch("src.annotation.annotation.import_atlas")
@patch("src.annotation.annotation.group_nodes")
@patch("src.annotation.annotation.find_fragment_matches")
@patch("src.annotation.annotation.build_molecular_families")
@patch("src.annotation.annotation.write_cx")
@patch("src.annotation.annotation.prepare_directory")
@patch("shutil.rmtree")
def test_fragment_snapms(
    mock_rmtree,
    mock_prepare_dir,
    mock_write_cx,
    mock_build_families,
    mock_find_matches,
    mock_group_nodes,
    mock_import_atlas,
    mock_read_cx,
    mock_annotation_config,
):
    mock_graph = nx.Graph()
    mock_read_cx.return_value = mock_graph
    mock_group_nodes.return_value = {"group_1": {"node_1"}}

    sub_graph = nx.Graph()
    sub_graph.add_node("node_1")
    mock_build_families.return_value = sub_graph

    fragment_snapms(mock_annotation_config)
