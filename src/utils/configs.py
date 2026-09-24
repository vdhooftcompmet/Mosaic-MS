from dataclasses import dataclass


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
