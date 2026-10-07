from dataclasses import dataclass, fields
from pathlib import Path

REPO = Path(__file__).parent.parent.parent.resolve()


def _resolve_path(path_str: str) -> str:
    """Resolve path against REPO if relative; return as-is if already absolute or empty."""
    if not path_str:
        return ""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(REPO / p)


def _print_formatted_config(header_title: str, instance: object) -> None:
    """Helper function to print dataclass fields formatted with bullets and borders in order of definition."""
    cls_fields = fields(instance)

    # Calculate maximum label width dynamically for clean alignment
    max_label_len = max(len(f.name.replace("_", " ").title()) for f in cls_fields)

    print("=" * 60)
    print(f"{header_title}:")
    for f in cls_fields:
        label = f.name.replace("_", " ").title()
        val = getattr(instance, f.name)
        padding = " " * (max_label_len - len(label))
        print(f"  • {label}:{padding} {val}")
    print("=" * 60)


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
        self.mgf = _resolve_path(self.mgf)
        self.motifs_path = _resolve_path(self.motifs_path)
        self.model_path = _resolve_path(self.model_path)

    def display(self) -> None:
        """Display MS2LDA configuration parameters."""
        _print_formatted_config("MS2LDA Configured Options", self)


@dataclass
class AddMS2LDAConfig:
    model: str
    graph: str
    threshold: float

    def __post_init__(self):
        self.model = _resolve_path(self.model)
        self.graph = _resolve_path(self.graph)

    def display(self) -> None:
        """Display Add MS2LDA configuration parameters."""
        _print_formatted_config("Add MS2LDA Configured Options", self)


@dataclass
class AddMAGConfig:
    motifs: str
    spec2vec_model_path: str
    library: str

    threshold: float
    cluster_delta: float
    criterium: str

    def __post_init__(self):
        self.motifs = _resolve_path(self.motifs)
        self.spec2vec_model_path = _resolve_path(self.spec2vec_model_path)
        self.library = _resolve_path(self.library)

    def display(self) -> None:
        """Display Add MAG configuration parameters."""
        _print_formatted_config("Add Motif Annotation Guidance (MAG) Configured Options", self)


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
        self.graph = _resolve_path(self.graph)
        self.reference_db = _resolve_path(self.reference_db)
        self.result_folder = _resolve_path(self.result_folder)

    def display(self) -> None:
        """Display SNAPMS configuration parameters."""
        _print_formatted_config("SNAPMS Configured Options", self)


@dataclass
class AddSNAPMSConfig:
    graph: str
    snapms: str

    def __post_init__(self):
        self.graph = _resolve_path(self.graph)
        self.snapms = _resolve_path(self.snapms)

    def display(self) -> None:
        """Display Add SNAPMS configuration parameters."""
        _print_formatted_config("Add SNAPMS Configured Options", self)


@dataclass
class MNConfig:
    mgf: str
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
        self.mgf = _resolve_path(self.mgf)
        self.cache_folder = _resolve_path(self.cache_folder)
        self.base_graph_path = _resolve_path(self.base_graph_path)
        self.threshold_graph_path = _resolve_path(self.threshold_graph_path)
        self.rescued_graph_path = _resolve_path(self.rescued_graph_path)

        if self.ms2deepscore_model_path:
            self.ms2deepscore_model_path = _resolve_path(self.ms2deepscore_model_path)
        if self.spec2vec_model_path:
            self.spec2vec_model_path = _resolve_path(self.spec2vec_model_path)

    def display(self) -> None:
        """Display MN configuration parameters."""
        _print_formatted_config("Molecular Networking Configured Options", self)
