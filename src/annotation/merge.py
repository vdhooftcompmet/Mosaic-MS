import logging

import networkx as nx
import numpy as np
import tomotopy as tp
from matchms import Spectrum
from matchms.similarity.FlashSimilarity import FlashSimilarity

from src.mn.mn import build_graph, filter_base_strategy
from src.ms2lda.ms2lda import extract_motifs
from src.utils.context import suppress_output
from src.utils.cx import write_cx

logging.getLogger("matchms").setLevel(logging.ERROR)


def merge_motifs(model_paths: list[str]) -> tuple[list[Spectrum], nx.Graph]:
    models = [tp.LDAModel.load(str(p)) for p in model_paths]
    all_motifs = []

    for model in models:
        model_motifs = extract_motifs(model, top_n_words=50, charge=1)
        k = len(model_motifs)
        for i, model_motif in enumerate(model_motifs):
            model_motif.set("topic_index", i)
            model_motif.set("model_k", k)
            all_motifs.append(model_motif)

    with suppress_output():
        modified_motifs = _losses_to_peaks(all_motifs)
        obj = FlashSimilarity(
            score_type="cosine", matching_mode="fragment", tolerance=0.01
        )
        similarity = obj.matrix(modified_motifs, modified_motifs)

    strat = filter_base_strategy(0.9)
    support = np.zeros_like(similarity)
    G = build_graph(all_motifs, similarity, support, strat)

    for i, m in enumerate(all_motifs):
        i_max = np.argmax(m.peaks.intensities)
        primary_mz = float(m.peaks.mz[i_max])
        G.nodes[i]["primary_mz"] = f"{primary_mz:.2f}"

    motif_clusters = list(nx.connected_components(G))
    motif_clusters = list(sorted(motif_clusters, key=len, reverse=True))

    result: list[Spectrum] = []

    for i, nodes in enumerate(motif_clusters):
        if len(nodes) <= len(models) * 0.75:
            break

        cluster_motifs = [all_motifs[n] for n in nodes]
        motif = _merge_spectra(cluster_motifs)
        motif.set("motif_id", i)
        result.append(motif)

    return result, G


def _merge_spectra(spectra: list[Spectrum], tolerance=0.01, normalize=True) -> Spectrum:
    assert len(spectra) != 0

    mz = np.concat([s.peaks.mz for s in spectra])
    intensities = np.concat([s.peaks.intensities for s in spectra])

    assert len(mz) != 0

    indices = np.argsort(mz)
    mz = mz[indices]
    intensities = intensities[indices]

    gaps = np.diff(mz) > tolerance
    split_indices = np.where(gaps)[0] + 1

    mz_bins = np.split(mz, split_indices)
    intensity_bins = np.split(intensities, split_indices)

    binned_mz = np.array([np.mean(b) for b in mz_bins])
    binned_intensities = np.array([np.sum(b) for b in intensity_bins])

    if normalize:
        binned_intensities = _normalize_intensities(binned_intensities)

    assert len(binned_mz) != 0

    return Spectrum(binned_mz, binned_intensities)


def _normalize_intensities(intensities: np.ndarray) -> np.ndarray:
    if len(intensities) == 0:
        return intensities

    max_intensity = intensities.max()
    if max_intensity == 0:
        return intensities
    return intensities / max_intensity


LOSSES_OFFSET = 2000


def _losses_to_peaks(spectra: list[Spectrum]) -> list[Spectrum]:
    result = []
    for spectrum in spectra:
        mz = spectrum.peaks.mz[spectrum.peaks.mz < LOSSES_OFFSET]
        i = spectrum.peaks.intensities[spectrum.peaks.mz < LOSSES_OFFSET]

        mz_pos, mz_neg = mz[mz > 0], mz[mz <= 0]
        i_pos, i_neg = i[mz > 0], i[mz <= 0]

        mz = np.concat([mz_pos, np.abs(mz_neg)[::-1] + LOSSES_OFFSET])
        i = np.concat([i_pos, i_neg[::-1]])
        result.append(Spectrum(mz, i, spectrum.metadata))
    return result
