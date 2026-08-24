import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from matchms.exporting import save_as_mgf

from scipy.sparse import save_npz, load_npz, csr_matrix
from mn.bootstrap import *
from mn.network import *
from utils.folders import prepare_directory
from utils.constants import *
from utils.cx import write_cx
from setup.paths import MN_STYLE_FILE

GRAPH_FUNCTIONS = {
    "base"      : base_graph, 
    "threshold" : threshold_graph, 
    "rescued"   : rescued_graph
}


def main(params: Namespace) -> None:
    if not Path(params.folder).exists():
        prepare_directory(params.folder)

    if Path(params.mgf_cleaned).exists():
        print("loading cleaned data...")
        spectra = list(load_from_mgf(params.mgf_cleaned))

    else:
        print("> cleaning spectra...")
        spectra = list(clean_mgf(params.mgf))
        
        print("> saving cleaned spectral data...")
        cleaned_path = Path(params.mgf_cleaned)
        save_as_mgf(spectra, str(cleaned_path), file_mode="w")

    similarity, support = load_similarity(spectra, params)

    graph_types = ["base", "threshold", "rescued"]
    file_names  = [params.base_graph_path, params.threshold_graph_path, params.rescued_graph_path]

    for graph_type, file_name in zip(graph_types, file_names):
        graph_fn = GRAPH_FUNCTIONS[graph_type]
        graph = graph_fn(similarity, support, spectra, params.similarity_type, params)

        add_cluster_numbering(graph)

        write_cx(graph, file_name, MN_STYLE_FILE)


def load_similarity(spectra: list[Spectrum], params: Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    avg_sim_file    = Path(params.folder) / params.average_similarity_file
    sim_file        = Path(params.folder) / params.similarity_file
    sup_file        = Path(params.folder) / params.support_file
    
    if avg_sim_file.exists() and sim_file.exists() and sup_file.exists() and params.cache_similarity:

        average_similarity = load_npz(avg_sim_file).toarray()
        similarity = load_npz(sim_file).toarray()
        support = load_npz(sup_file).toarray()

    else:
        similarity = plain_similarity(spectra, params)
        save_npz(sim_file, csr_matrix(similarity))

        average_similarity, support = calculate_bootstrapping(spectra, params)
        
        save_npz(avg_sim_file, csr_matrix(average_similarity))
        save_npz(sup_file, csr_matrix(support))

    if params.use_average_similarity:
        return average_similarity, support
    else:
        return similarity, support