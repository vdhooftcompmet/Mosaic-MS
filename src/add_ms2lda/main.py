from pathlib import Path
import numpy as np
from matchms import Spectrum
from argparse import Namespace
import tomotopy as tp
from tqdm import tqdm
from matchms.importing import load_from_mgf
from ms2lda.preprocessing import spectra_to_documents
from utils.cx import read_cx, write_cx
from setup.paths import MN_STYLE_FILE
import ast


def main(params) -> None:
    model_path = Path(params.model).resolve()
    
    assert model_path.exists(), f"Error: Model file does not exist at {model_path}"
    assert model_path.is_file(), f"Error: {model_path} is a directory, not a file!"

    mn = read_cx(params.graph)

    spectra = []
    for node in mn:
        spectrum_data_str = str(mn.nodes[node]["peaks_json"])
        spectrum_data = to_list(spectrum_data_str)
        
        mz = np.array([x[0] for x in spectrum_data])
        i  = np.array([x[1] for x in spectrum_data])
        metadata = {k: v for k, v in mn.nodes[node].items() if k != "peaks_json"}
        metadata = {k: v for k, v in metadata.items() if v != None}
        metadata["retention_time"]  = metadata.get("rtinminutes")
        metadata["retention_index"] = 0
            
        spectrum = Spectrum(mz, i, metadata)
        spectra.append(spectrum)
        
    model = tp.LDAModel.load(str(model_path))
    topic_words = get_topic_words(model)
    params.dataset_significant_digits = derive_significant_digits(topic_words)
    params.dataset_acquisition_type = dataset_acquisition_type(topic_words)

    result = run_overlap_scores_calculation(spectra, model, params)
    _beta_matrix, _phi_matrix, _theta_matrix, overlap_scores = result

    threshold = float(params.threshold)

    

    assert len(mn) == overlap_scores.shape[1], f"{len(mn)} vs {overlap_scores.shape[1]}"

    for node in mn:
        i = int(node)
        scores = overlap_scores[:, i]

        present_motifs = [str(m) for m, s in enumerate(scores) if s > threshold]
        mn.nodes[node]["motifs"] = ";".join(present_motifs)

        for ii, score in enumerate(scores):
            mn.nodes[node][f"motif_{ii}"] = 1 if score > threshold else 0

    write_cx(mn, params.graph, MN_STYLE_FILE)


def to_list(list_str):
    return ast.literal_eval(list_str)
    

def run_overlap_scores_calculation(spectra: list[Spectrum],  model: tp.LDAModel,  params: Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # see original/MS2LDA/Visualization/lda_dict for original function

    spectra = [s for s in spectra]

    n_words = len(model.used_vocabs)
    n_docs  = len(spectra)
    n_topics = model.k

    words_per_spectra = spectra_to_documents(spectra, params)

    documents = []
    for words in tqdm(words_per_spectra):
        words = list(words)

        if not words:
            documents.append(None)
        else:
            doc = model.make_doc(list(words))
            model.infer(doc)  
            documents.append(doc)

    beta_matrix = np.zeros((n_topics, n_words), dtype=float)  # Beta signifies the distribution of words for every topic
    for i in range(n_topics):
        word_probs = model.get_topic_word_dist(i)
        beta_matrix[i, :] = np.array(word_probs)

    theta_matrix = np.zeros((n_topics, n_docs))  # Theta signifies the presence of each topic for every doc
    for i, doc in enumerate(documents):

        if doc is None:
            continue

        theta_matrix[:, i] = doc.get_topic_dist() 

    word_counts = np.zeros((n_words, n_docs), dtype=np.int32)  # Word counts is a simple histogram that couns how mane times the same word occurs in the same doc
    for i, doc in enumerate(documents):

        if doc is None:
            continue

        doc_indices = np.array([w for w in doc.words], dtype=np.int32)
        valid_indices = doc_indices[doc_indices < n_words]
        
        np.add.at(word_counts[:, i], valid_indices, 1)

    phi_matrix = beta_matrix @ word_counts   # tw * wd -> td; Phi signifies the topic distribution in every doc 
    phi_matrix /= phi_matrix.sum(axis=0, keepdims=True) + 1e-12  # Phi is noralized for topic size to make a fair comparison

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
        if '@' in word:
            mass_str = word.split('@')[1]
    
            # Count decimal places (digits after '.')
            dec_count = len(mass_str.split('.')[1]) if '.' in mass_str else 0
            decimal_places.add(dec_count)
    
            # Count total significant digits (digits excluding '.' and leading zeros)
            clean_num = mass_str.replace('.', '').lstrip('0')
            sig_digits.add(len(clean_num))

    return max(decimal_places)
    

def dataset_acquisition_type(topic_words):
    has_losses = any(word.startswith('loss@') for word in topic_words)

    if has_losses:
        return "DDA"
    return "DIA"

    