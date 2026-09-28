from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.resolve()


@dataclass
class MS2LDAConfig:
    mgf: str
    motifs_path: str
    model_path: str

    nr_of_motifs: int
    top_n_words: int
    iterations: int

    dataset_acquisition_type: str
    dataset_charge: int
    dataset_significant_digits: int

    train_parallel: int
    train_workers: int

    model_rm_top: int
    model_min_cf: int
    model_min_df: int
    model_alpha: float
    model_eta: float
    model_seed: int

    conv_step_size: int
    conv_window_size: int
    conv_threshold: float
    conv_type: str

    prep_min_mz: int
    prep_max_mz: int
    prep_max_frags: int
    prep_min_frags: int
    prep_min_intensity: float
    prep_max_intensity: float

    def __post_init__(self):
        self.mgf = str(REPO / self.mgf)
        self.motifs_path = str(REPO / self.motifs_path)
        self.model_path = str(REPO / self.model_path)


@dataclass
class SNAPMSConfig:
    graph: str
    reference_db: str
    result_folder: str

    ppm_error: int
    cutoff: float
    adduct_list: list

    min_cluster_size: int
    max_cluster_size: int
    min_annotation_size: int

    remove_duplicates: bool
    detect_adduct: bool

    def __post_init__(self):
        self.graph = str(REPO / self.graph)
        self.reference_db = str(REPO / self.reference_db)
        self.result_folder = str(REPO / self.result_folder)


@dataclass
class MNConfig:
    mgf: str
    folder: str
    cache_folder: str
    similarity_type: str
    base_graph_path: str
    threshold_graph_path: str
    rescued_graph_path: str

    similarity_threshold: float
    support_threshold: float
    max_component_size: int
    rescue_similarity_threshold: float

    use_average_similarity: bool
    cache_similarity: bool

    binning_decimals: int
    B: int
    k: int
    seed: int
    flash_tolerance: float

    ms2deepscore_model_path: str = ""
    spec2vec_model_path: str = ""

    def __post_init__(self):
        self.mgf = str(REPO / self.mgf)
        self.folder = str(REPO / self.folder)
        self.cache_folder = str(REPO / self.cache_folder)
        self.base_graph_path = str(REPO / self.base_graph_path)
        self.threshold_graph_path = str(REPO / self.threshold_graph_path)
        self.rescued_graph_path = str(REPO / self.rescued_graph_path)

        self.ms2deepscore_model_path = str(REPO / self.ms2deepscore_model_path)
        self.spec2vec_model_path = str(REPO / self.spec2vec_model_path)
