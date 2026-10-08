import logging
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import gensim
import numpy as np
from matchms import Spectrum
from matchms.exporting import save_as_mgf
from matchms.filtering import normalize_intensities
from matchms.importing import load_from_mgf
from rdkit import Chem
from sklearn.cluster import AgglomerativeClustering
from spec2vec import Spec2Vec
from spec2vec.vector_operations import cosine_similarity_matrix

from src.utils.configs import AddMAGConfig
from src.utils.progress_bar import track
from src.utils.substructure import greedy_substructure_finder

logging.getLogger("matchms").setLevel(logging.ERROR)

LibraryMatch = namedtuple("LibraryMatch", ["spectrum", "score"])
Doc = namedtuple("Doc", ["words", "weights"])


def main(config: AddMAGConfig) -> None:
    motifs = list(load_from_mgf(str(config.motifs)))
    w2v = gensim.models.Word2Vec.load(str(config.spec2vec_model_path))
    model = Spec2Vec(
        model=w2v,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=100.0,
        progress_bar=False,
    )
    library_path = str(Path(config.library))

    all_matches = find_library_matches(motifs, model, library_path, config.threshold)

    print(
        f"Found matches for {sum(bool(x) for x in all_matches)}/{len(all_matches)} motifs"
    )

    for motif, matches in zip(motifs, all_matches):
        cluster = select_matches(
            matches, motif, model, config.cluster_delta, config.criterium
        )
        mols = []
        for match in cluster:
            spectrum = match.spectrum
            smiles = spectrum.get("smiles") or spectrum.get("SMILES")
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)

        substructures = Counter()
        for frag1, frag2 in greedy_substructure_finder(mols):
            for frag in [frag1, frag2]:
                smiles = Chem.MolToSmiles(frag)
                substructures[smiles] += 1

        most_common_substructure = substructures.most_common(1)

        if not most_common_substructure:
            motif.set("mag", "")
            continue

        mcs, _count = most_common_substructure[0]
        motif.set("mag", mcs)

    save_as_mgf(motifs, str(config.motifs), file_mode="w")


def find_library_matches(
    motifs: list[Spectrum],
    model: Spec2Vec,
    library_path: str | Path,
    threshold: float,
    batch_size: int = 1024,
) -> list[list[LibraryMatch]]:

    result = defaultdict(list)
    total = count_mgf_spectra(str(library_path))
    total = total // batch_size + (1 if total % batch_size != 0 else 0)
    generator = load_from_mgf(str(library_path))

    motif_docs = [to_doc(m, model) for m in motifs]
    motif_vectors = [convert_to_vector(model, d) for d in motif_docs]
    motif_vectors = np.array(motif_vectors)

    for batch in track(
        make_batches(generator, batch_size),
        total=total,
        description="finding library matches...",
    ):
        batch = [normalize_intensities(s) for s in batch]
        batch = [s for s in batch if s is not None]

        batch_docs = [to_doc(m, model) for m in batch]
        batch_vectors = [convert_to_vector(model, d) for d in batch_docs]
        batch_vectors = np.array(batch_vectors)
        similarity = cosine_similarity_matrix(motif_vectors, batch_vectors)

        hits = np.argwhere(similarity > threshold)
        for a, b in hits:
            hit: Spectrum = batch[b].clone()
            score = similarity[a, b]
            match = LibraryMatch(hit, score)
            result[a].append(match)

    result = [result[i] for i, _ in enumerate(motifs)]
    return result


def count_mgf_spectra(input_path: Path | str) -> int:
    input_path = Path(input_path)
    count = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().upper().startswith("BEGIN IONS"):
                count += 1
    return count


def make_batches(generator, size):
    batch = []
    batch_size = 0

    for spectrum in generator:
        batch.append(spectrum)
        batch_size += 1

        if batch_size == size:
            yield batch
            batch = []
            batch_size = 0

    if batch:
        yield batch


def select_matches(
    matches: list[LibraryMatch],
    motif: Spectrum,
    model: Spec2Vec,
    cluster_delta: float,
    criterium: str,
):
    if not matches:
        return []

    motif_vectors = []
    doc = to_doc(motif, model)
    for masked_doc in mask_doc(doc):
        motif_vector = convert_to_vector(model, masked_doc)
        motif_vectors.append(motif_vector)

    matches_vectors = []
    for m in matches:
        match_vector = convert_to_vector(model, to_doc(m.spectrum, model))
        matches_vectors.append(match_vector)

    motif_vectors = np.array(motif_vectors)
    matches_vectors = np.array(matches_vectors)
    similarity = cosine_similarity_matrix(motif_vectors, matches_vectors).T
    labels = agglomerative_clustering(similarity, cluster_delta)

    if criterium == "best":
        best_hit = labels[0]
        indices = np.argwhere(labels == best_hit).flatten()
    elif criterium == "biggest":
        counts = np.bincount(labels)
        biggest_label = np.argmax(counts)
        indices = np.argwhere(labels == biggest_label).flatten()
    else:
        raise ValueError()

    selected_matches = [matches[i] for i in indices]
    return selected_matches


def to_doc(spectrum: Spectrum, model: Spec2Vec) -> Doc:
    mz, intensities = [], []

    losses = spectrum.losses
    if losses is not None:  # convert the losses, add them as negative mz-values
        mz += list(losses.mz * -1)[::-1]
        intensities += list(losses.intensities)[::-1]

    # this order keeps mz values sorted from small to large
    mz += list(spectrum.peaks.mz)
    intensities += list(spectrum.peaks.intensities)

    words = np.array([to_word(x, model) for x in mz], dtype=str)
    weights = np.array(intensities)
    return Doc(words, weights)


def to_word(mz: float, model: Spec2Vec):
    if mz >= 0:
        return f"peak@{mz:.{model.n_decimals}f}"

    mz = abs(mz)
    return f"loss@{mz:.{model.n_decimals}f}"


def mask_doc(doc: Doc):
    words, weights = list(doc.words), list(doc.weights)
    for i in range(len(words)):
        masked_words = words[:i] + words[i + 1 :]
        masked_weights = weights[:i] + weights[i + 1 :]
        yield Doc(np.array(masked_words), np.array(masked_weights))


def convert_to_vector(model: Spec2Vec, doc: Doc) -> np.ndarray:
    words = np.array(doc.words)
    weights = np.array(doc.weights)

    weights = normalize_vector(weights)

    mask = np.array([word in model.model.wv.key_to_index for word in words])
    words = words[mask]
    weights = weights[mask]

    if len(words) == 0:
        return np.zeros(model.model.wv.vector_size)

    weights = weights.reshape(len(words), 1)
    word_vectors = model.model.wv[words]
    weights_raised = np.power(weights, model.intensity_weighting_power)
    weights_raised_tiled = np.tile(weights_raised, (1, model.model.wv.vector_size))
    return np.sum(word_vectors * weights_raised_tiled, 0)


def normalize_vector(vector):
    max_value = np.max(np.abs(vector))
    if max_value != 0:
        vector /= max_value
    return vector


def agglomerative_clustering(masked_spectra_similarity: np.ndarray, delta=0.6):
    if masked_spectra_similarity.shape[0] <= 1:
        return np.array([0])

    cosine_distance = 1 - delta
    cosine_distance_matrix = 1 - masked_spectra_similarity
    clustering = AgglomerativeClustering(
        distance_threshold=cosine_distance,
        n_clusters=None,
        linkage="complete",
    )
    labels = clustering.fit_predict(cosine_distance_matrix)
    return labels
