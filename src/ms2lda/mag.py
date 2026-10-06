from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import gensim
import numpy as np
from matchms import Spectrum
from matchms.importing import load_from_mgf
from rdkit import Chem
from sklearn.cluster import AgglomerativeClustering
from spec2vec import Spec2Vec
from spec2vec.vector_operations import cosine_similarity_matrix

LibraryMatch = namedtuple("LibraryMatch", ["spectrum", "score"])
Doc = namedtuple("Doc", ["words", "weights"])


def main(config):
    result = []
    motifs = list(load_from_mgf(str(config.motifs)))
    w2v = gensim.models.Word2Vec.load(str(config.spec2vec_model_path))
    model = Spec2Vec(
        model=w2v,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=20.0,
        progress_bar=False,
    )
    library_path = str(Path(config.library))

    all_matches = find_library_matches(motifs, model, library_path, config.threshold)
    for motif, matches in zip(motifs, all_matches):
        cluster = select_matches(
            matches, motif, model, config.cluster_delta, config.criterium
        )
        spectra = [m.spectrum for m in cluster]
        smiles = [s.get("smiles") or s.get("SMILES") for s in spectra]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        mols = [m for m in mols if m is not None]
        counter = greedy_substructure_finder(mols)
        most_common_substructure = counter.most_common(1)

        if not most_common_substructure:
            result.append(None)
            continue

        mcs, _count = most_common_substructure[0]
        result.append(mcs)
    return result


def find_library_matches(
    motifs: list[Spectrum],
    similarity_metric,
    library_path: str | Path,
    threshold: float,
    batch_size: int = 1024,
) -> list[list[LibraryMatch]]:

    result = defaultdict(list)
    generator = load_from_mgf(str(library_path))

    for batch in make_batches(generator, batch_size):
        similarity = similarity_metric.matrix(motifs, batch)
        hits = np.argwhere(similarity > threshold)
        for a, b in hits:
            hit: Spectrum = batch[b].clone()
            score = similarity[a, b]
            match = LibraryMatch(hit, score)
            result[a].append(match)

    result = [result[i] for i, _ in enumerate(motifs)]
    return result


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


def greedy_substructure_finder(mols):
    substructures = Counter()
    for mol in mols:
        num_bonds = mol.GetNumBonds()
        bonds_index = list(range(num_bonds))

        for bond_index in bonds_index:
            fragments = generate_substructure(mol, bond_index)

            if fragments is None:
                continue

            smiles1, smiles2 = fragments
            substructures[smiles1] += 1
            substructures[smiles2] += 1

    return substructures


def generate_substructure(mol, bond_index: int) -> None | tuple[str, str]:
    with Chem.RWMol(mol) as rwmol:
        selected_bond = rwmol.GetBondWithIdx(bond_index)

        is_not_single_bond = selected_bond.GetBondType() != Chem.BondType.SINGLE
        is_in_ring = selected_bond.IsInRing()

        if is_not_single_bond or is_in_ring:
            return None

        u = selected_bond.GetBeginAtomIdx()
        v = selected_bond.GetEndAtomIdx()
        rwmol.RemoveBond(u, v)

    frags = Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False)
    if len(frags) != 2:
        return None

    smiles1 = Chem.MolToSmiles(frags[0])
    smiles2 = Chem.MolToSmiles(frags[1])

    return smiles1, smiles2
