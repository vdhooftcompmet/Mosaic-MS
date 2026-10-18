import json
import logging

import numpy as np
from matchms import Spectrum
from matchms.importing import load_from_mgf

from src.ms2lda.ms2lda import spectra_to_documents
from src.utils.configs import AddMS2LDAConfig
from src.utils.cx import read_cx, write_cx
from src.utils.paths import MN_STYLE_FILE
from src.utils.progress_bar import track

logging.getLogger("matchms").setLevel(logging.ERROR)


def add_ms2lda(params: AddMS2LDAConfig) -> None:
    import numpy as np
    from matchms import Spectrum

    motifs = list(load_from_mgf(params.motifs))

    mn = read_cx(params.graph)

    spectra = []
    for node in mn:
        spectrum_data_str = str(mn.nodes[node]["peaks_json"])
        spectrum_data = parse_spectrum_peaks(spectrum_data_str)

        mz = np.array([x[0] for x in spectrum_data])
        i = np.array([x[1] for x in spectrum_data])
        metadata = {k: v for k, v in mn.nodes[node].items() if k != "peaks_json"}
        metadata = {k: v for k, v in metadata.items() if v is not None}
        metadata["retention_time"] = metadata.get("rtinminutes")
        metadata["retention_index"] = 0

        spectrum = Spectrum(mz, i, metadata)
        spectra.append(spectrum)

    overlap_scores = _calc_overlap_scores(motifs, spectra, True, 2)
    threshold = float(params.threshold)

    print(f"{len(motifs) = }")
    assert len(mn) == overlap_scores.shape[1], f"{len(mn)} vs {overlap_scores.shape[1]}"

    node_list = list(mn.nodes())
    for i, node in enumerate(node_list):
        scores = overlap_scores[:, i]
        present_motifs = [str(m) for m, s in enumerate(scores) if s > threshold]
        mn.nodes[node]["motifs"] = ";".join(present_motifs)

        for ii, score in enumerate(scores):
            mn.nodes[node][f"motif_{ii}"] = 1 if score > threshold else 0

    write_cx(mn, params.graph, MN_STYLE_FILE)


def parse_spectrum_peaks(peaks_raw: str | list) -> list:
    if isinstance(peaks_raw, str):
        return json.loads(peaks_raw)
    return peaks_raw


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
