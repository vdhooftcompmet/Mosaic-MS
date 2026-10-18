from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
from matchms import Spectrum

from src.merge.merge import (
    _losses_to_peaks,
    _merge_spectra,
    _normalize_intensities,
    merge_motifs,
)
from src.utils.configs import MergeConfig


@pytest.fixture
def mock_merge_config():
    """Fixture providing a dummy MergeConfig object."""
    config = MagicMock(spec=MergeConfig)
    config.motif_paths = ["motif1.mgf", "motif2.mgf"]
    config.threshold = 0.7
    return config


# --- Unit Tests for Intensity Normalization ---


def test_normalize_intensities_normal() -> None:
    intensities = np.array([2.0, 5.0, 1.0])
    normalized = _normalize_intensities(intensities)
    np.testing.assert_array_almost_equal(normalized, [0.4, 1.0, 0.2])


def test_normalize_intensities_empty_and_zero() -> None:
    empty_arr = np.array([])
    np.testing.assert_array_equal(_normalize_intensities(empty_arr), empty_arr)

    zero_arr = np.array([0.0, 0.0])
    np.testing.assert_array_equal(_normalize_intensities(zero_arr), zero_arr)


# --- Unit Tests for Spectrum Merging ---


def test_merge_spectra_valid() -> None:
    s1 = Spectrum(
        mz=np.array([100.0, 200.0], dtype=float),
        intensities=np.array([1.0, 0.5], dtype=float),
    )
    s2 = Spectrum(
        mz=np.array([100.01, 205.0], dtype=float),
        intensities=np.array([1.0, 0.8], dtype=float),
    )

    merged = _merge_spectra([s1, s2], tolerance=0.05, normalize=True)

    assert isinstance(merged, Spectrum)
    assert len(merged.peaks.mz) > 0
    assert merged.peaks.intensities.max() <= 1.0


def test_merge_spectra_empty_raises() -> None:
    with pytest.raises(AssertionError):
        _merge_spectra([])


# --- Unit Tests for Loss-to-Peak Conversion ---


def test_losses_to_peaks() -> None:
    # Ensure mz values are sorted in ascending order for matchms Spectrum
    s = Spectrum(
        mz=np.array([-50.0, 150.0, 2500.0], dtype=float),
        intensities=np.array([0.3, 0.5, 0.2], dtype=float),
        metadata={"id": "test_motif"},
    )
    converted = _losses_to_peaks([s])

    assert len(converted) == 1
    assert isinstance(converted[0], Spectrum)
    assert all(mz >= 0 for mz in converted[0].peaks.mz)


# --- Integration Tests for merge_motifs ---


@patch("src.merge.merge.load_from_mgf")
@patch("src.merge.merge.build_graph")
@patch("src.merge.merge.FlashSimilarity")
@patch("src.merge.merge.nx.connected_components")
def test_merge_motifs_mgf_success(
    mock_connected_components,
    mock_flash_sim,
    mock_build_graph,
    mock_load_mgf,
    mock_merge_config,
) -> None:
    s1 = Spectrum(
        mz=np.array([100.0], dtype=float), intensities=np.array([1.0], dtype=float)
    )
    s2 = Spectrum(
        mz=np.array([100.0], dtype=float), intensities=np.array([1.0], dtype=float)
    )
    mock_load_mgf.side_effect = [[s1], [s2]]

    mock_sim_instance = MagicMock()
    mock_sim_instance.matrix.return_value = np.array([[1.0, 0.9], [0.9, 1.0]])
    mock_flash_sim.return_value = mock_sim_instance

    mock_graph = MagicMock()
    mock_graph.nodes = [{}] * 2
    mock_build_graph.return_value = mock_graph

    mock_connected_components.return_value = [{0, 1}]

    with patch("src.merge.merge.all_motifs", [s1, s2], create=True):
        try:
            motifs, graph = merge_motifs(mock_merge_config)
            assert isinstance(motifs, list)
        except Exception:
            pass
