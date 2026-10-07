import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import tomotopy as tp
from matchms import Spectrum

from src.ms2lda.ms2lda import spectra_to_documents
from src.utils.configs import AddMS2LDAConfig
from src.utils.cx import read_cx, write_cx
from src.utils.paths import MN_STYLE_FILE
from src.utils.progress_bar import track
from src.utils.motifs import run_overlap_scores_calculation


def main(params: AddMS2LDAConfig) -> None:
    model_path = Path(params.model).resolve()

    assert model_path.exists(), f"Error: Model file does not exist at {model_path}"
    assert model_path.is_file(), f"Error: {model_path} is a directory, not a file!"

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

    model = tp.LDAModel.load(str(model_path))
    topic_words = get_topic_words(model)

    namespace = Namespace(**vars(params))
    namespace.dataset_significant_digits = derive_significant_digits(topic_words)
    namespace.dataset_acquisition_type = derive_dataset_acquisition_type(topic_words)

    result = run_overlap_scores_calculation(spectra, model, namespace)
    _beta_matrix, _phi_matrix, _theta_matrix, overlap_scores = result

    threshold = float(params.threshold)

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
