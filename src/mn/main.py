from scipy.sparse import save_npz, load_npz, csr_matrix
from mn.bootstrap import *
from mn.network import *
from utils.folders import prepare_directory
from utils.constants import *
from setup.paths import MN_STYLE_FILE
import hashlib
from matchms.importing import load_from_mgf
from matchms import SpectrumProcessor
from utils.cx import write_cx
from matchms.filtering.default_pipelines import DEFAULT_FILTERS, CLEAN_PEAKS
from utils.configs import MNConfig


GRAPH_FUNCTIONS = {
    "base"      : base_graph,
    "threshold" : threshold_graph,
    "rescued"   : rescued_graph
}


def main(config: MNConfig) -> None:
    if not Path(config.folder).exists():
        prepare_directory(config.folder)

    spectra = clean_mgf(config.mgf)
    similarity, support = _similarity_cache(calculate_bootstrapping)(spectra, config)

    file_names = {
        "base"      : config.base_graph_path, 
        "threshold" : config.threshold_graph_path, 
        "rescued"   : config.rescued_graph_path, 
    }

    for graph_type, file_name in file_names.items():
        graph_fn = GRAPH_FUNCTIONS[graph_type]
        graph = graph_fn(similarity, support, spectra, config.similarity_type, config)

        add_cluster_numbering(graph)

        write_cx(graph, file_name, MN_STYLE_FILE)


def _similarity_cache(fn):
    def inner(spectra, config: MNConfig):
        files = {}
        for file_type in ["avg_sim", "tot_sim", "tot_sup"]:
            files[file_type] = Path(config.cache_folder) / _make_cache_name(file_type, config)
    
        if all(f.exists() for f in files.values()):
            data = {data_type: load_npz(f).toarray() for data_type, f in files.items()}
        else:
            data = {}
            data["tot_sim"] = plain_similarity(spectra, config)
            data["avg_sim"], data["tot_sup"] = fn(spectra, config)
    
            if config.cache_similarity:
                for file_type, file_name in files.items():
                    matrix = csr_matrix(data[file_type])
                    save_npz(str(file_name), csr_matrix(matrix))
    
        if config.use_average_similarity:
            return data["avg_sim"], data["tot_sup"]
        else:
            return data["tot_sim"], data["tot_sup"]

    return inner


def _make_cache_name(file_type: str, config: MNConfig) -> str:
    hasher = hashlib.sha256()

    with open(str(config.mgf), "rb") as f:
        while chunk := f.read(65536): 
            hasher.update(chunk)

    hasher.update(str(config.similarity_type).lower().encode("utf-8"))
    hasher.update(str(file_type)             .lower().encode("utf-8"))
    hasher.update(str(config.B)              .lower().encode("utf-8"))
    hasher.update(str(config.seed)           .lower().encode("utf-8"))
    hexadecimal_string = hasher.hexdigest()
    file_name = f"{config.similarity_type}-{hexadecimal_string[:20]}.npz"
    return file_name


def clean_mgf(path: Path | str):
    assert isinstance(path, (str, Path)), "path must be a Path object or a string"

    spectra = list(load_from_mgf(path))
    spectrum_processor = SpectrumProcessor(DEFAULT_FILTERS + CLEAN_PEAKS)
    result, _ = spectrum_processor.process_spectra(spectra, progress_bar=False)

    for i, spectrum in enumerate(result):
        spectrum.set(KEY_SPECTRUM_ID, i)

    return result
        


