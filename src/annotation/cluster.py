import logging

import networkx as nx
import numpy as np
from matchms import Spectrum

from src.ms2lda.ms2lda import spectra_to_documents
from src.utils.progress_bar import track

logging.getLogger("matchms").setLevel(logging.ERROR)


def build_spectra_graph(motifs: list[Spectrum], spectra: list[Spectrum]) -> nx.Graph:

    G = nx.Graph()

    for motif_index, spectrum_indices in enumerate(_detect_motifs(motifs, spectra)):
        motif = motifs[motif_index]
        motif_metadata = motif.to_dict()
        motif_metadata = {f"motif_{k}": str(v) for k, v in motif_metadata.items()}
        # add nodes
        for spectrum_index in spectrum_indices:
            spectrum = spectra[spectrum_index].clone()

            metadata = {k: str(v) for k, v in spectrum.to_dict().items()}

            # !! add cluster numbering based on motif index, not based on size
            metadata |= {"mn_cluster_id": motif_index}

            # !! we need motif metadata later for annotation
            metadata |= motif_metadata

            G.add_node(f"m{motif_index}s{spectrum_index}", **metadata)

        # add edges
        for u, spectrum_index_u in enumerate(spectrum_indices):
            for v, spectrum_index_v in enumerate(spectrum_indices):
                if u <= v:
                    continue

                id_u = f"m{motif_index}s{spectrum_index_u}"
                id_v = f"m{motif_index}s{spectrum_index_v}"
                G.add_edge(id_u, id_v)
    return G


def _detect_motifs(motifs: list[Spectrum], spectra: list[Spectrum]):
    overlap_scores = _calc_overlap_scores(motifs, spectra, True, 2)

    threshold = 0.01

    for motif_index in range(overlap_scores.shape[0]):
        scores = overlap_scores[motif_index, :]
        assert len(scores) == len(spectra)

        above_threshold = np.where(scores > threshold)[0]
        top_n = np.argsort(scores)[-10:]
        intersect = np.intersect1d(above_threshold, top_n)
        indices = [int(x) for x in intersect]
        yield indices


def _calc_overlap_scores(
    motifs: list[Spectrum],
    spectra: list[Spectrum],
    calc_losses: bool,
    significant_digits: int,
) -> np.ndarray:

    # 1. Convert both dataset spectra and consensus motifs into documents using your function
    spectra_docs = spectra_to_documents(spectra, calc_losses, significant_digits)
    motif_docs = spectra_to_documents(motifs, calc_losses, significant_digits)

    # 2. Build a unified vocabulary from all unique words found across both sets
    all_words = set()
    for doc in spectra_docs + motif_docs:
        if doc:
            all_words.update(doc)
    used_vocabs = sorted(list(all_words))
    word_to_idx = {word: idx for idx, word in enumerate(used_vocabs)}

    n_words = len(used_vocabs)
    n_docs = len(spectra)
    n_topics = len(motifs)

    # 3. Build beta_matrix (Topics x Words) from the consensus motif documents
    beta_matrix = np.zeros((n_topics, n_words), dtype=float)
    for i, doc in enumerate(motif_docs):
        if not doc:
            continue
        for word in doc:
            if word in word_to_idx:
                beta_matrix[i, word_to_idx[word]] += 1

    # Normalize each topic row to sum to 1 (proper probability distribution)
    row_sums = beta_matrix.sum(axis=1, keepdims=True)
    beta_matrix = np.divide(
        beta_matrix, row_sums, out=np.zeros_like(beta_matrix), where=row_sums != 0
    )

    # 4. Build word_counts matrix (Words x Docs) for the dataset spectra
    word_counts = np.zeros((n_words, n_docs), dtype=np.int32)
    for j, doc in enumerate(
        track(spectra_docs, description="processing spectra documents...")
    ):
        if not doc:
            continue
        for word in doc:
            if word in word_to_idx:
                word_counts[word_to_idx[word], j] += 1

    # 5. Calculate phi_matrix (Topics x Docs) -> topic distribution in every doc
    phi_matrix = beta_matrix @ word_counts
    phi_matrix /= phi_matrix.sum(axis=0, keepdims=True) + 1e-12

    # 6. Approximate theta_matrix (Topics x Docs) using vector projections (cosine similarity)
    doc_vectors = word_counts.astype(float)
    doc_norms = np.linalg.norm(doc_vectors, axis=0, keepdims=True) + 1e-12
    normalized_docs = doc_vectors / doc_norms

    motif_norms = np.linalg.norm(beta_matrix, axis=1, keepdims=True) + 1e-12
    normalized_motifs = beta_matrix / motif_norms

    theta_matrix = normalized_motifs @ normalized_docs
    theta_matrix = np.clip(theta_matrix, 0, None)  # Ensure non-negative
    theta_matrix /= theta_matrix.sum(axis=0, keepdims=True) + 1e-12

    # 7. Final overlap scores calculation
    overlap_scores = phi_matrix * theta_matrix

    return overlap_scores
