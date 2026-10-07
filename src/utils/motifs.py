import argparse

import numpy as np
import tomotopy as tp
from matchms import Spectrum

from src.ms2lda.ms2lda import spectra_to_documents
from src.utils.progress_bar import track


def run_overlap_scores_calculation(
    spectra: list[Spectrum], model: tp.LDAModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # see original/MS2LDA/Visualization/lda_dict for original function

    topic_words = get_topic_words(model)

    namespace = argparse.Namespace()
    namespace.dataset_significant_digits = derive_significant_digits(topic_words)
    namespace.dataset_acquisition_type = derive_dataset_acquisition_type(topic_words)

    spectra = [s for s in spectra]

    n_words = len(model.used_vocabs)
    n_docs = len(spectra)
    n_topics = model.k

    words_per_spectra = spectra_to_documents(spectra, namespace)

    documents = []
    for words in track(words_per_spectra, description="detecting motifs..."):
        words = list(words)

        if not words:
            documents.append(None)
        else:
            doc = model.make_doc(list(words))
            model.infer(doc)
            documents.append(doc)

    beta_matrix = np.zeros(
        (n_topics, n_words), dtype=float
    )  # Beta signifies the distribution of words for every topic
    for i in range(n_topics):
        word_probs = model.get_topic_word_dist(i)
        beta_matrix[i, :] = np.array(word_probs)

    theta_matrix = np.zeros(
        (n_topics, n_docs)
    )  # Theta signifies the presence of each topic for every doc
    for i, doc in enumerate(documents):
        if doc is None:
            continue

        theta_matrix[:, i] = doc.get_topic_dist()

    word_counts = np.zeros(
        (n_words, n_docs), dtype=np.int32
    )  # Word counts is a simple histogram that couns how mane times the same word occurs in the same doc
    for i, doc in enumerate(documents):
        if doc is None:
            continue

        doc_indices = np.array([w for w in doc.words], dtype=np.int32)
        valid_indices = doc_indices[doc_indices < n_words]

        np.add.at(word_counts[:, i], valid_indices, 1)

    phi_matrix = (
        beta_matrix @ word_counts
    )  # tw * wd -> td; Phi signifies the topic distribution in every doc
    phi_matrix /= (
        phi_matrix.sum(axis=0, keepdims=True) + 1e-12
    )  # Phi is normalized for topic size to make a fair comparison

    overlap_scores = phi_matrix * theta_matrix

    return beta_matrix, phi_matrix, theta_matrix, overlap_scores


def get_topic_words(model):
    all_words = []
    for topic_id in range(model.k):
        top_words = model.get_topic_words(topic_id, top_n=10)
        for word, value in top_words:
            all_words.append(word)
    return all_words


def derive_significant_digits(topic_words):
    decimal_places = set()
    sig_digits = set()

    for word in topic_words:
        if "@" in word:
            mass_str = word.split("@")[1]

            # Count decimal places (digits after '.')
            dec_count = len(mass_str.split(".")[1]) if "." in mass_str else 0
            decimal_places.add(dec_count)

            # Count total significant digits (digits excluding '.' and leading zeros)
            clean_num = mass_str.replace(".", "").lstrip("0")
            sig_digits.add(len(clean_num))

    return max(decimal_places, default=0)


def derive_dataset_acquisition_type(topic_words) -> str:
    has_losses = any(word.startswith("loss@") for word in topic_words)

    if has_losses:
        return "DDA"
    return "DIA"
