import argparse
import json
from pathlib import Path

from src.utils.cli import prepare_args
from src.utils.configs import AddMS2LDAConfig
from src.utils.cx import read_cx, write_cx
from src.utils.paths import MN_STYLE_FILE

REPO = Path(__file__).parent.parent.parent.resolve()

DEFAULT_ADD_MS2LDA_CONFIG = REPO / "config" / "add_ms2lda.yaml"


def main(args: argparse.Namespace):
    args = prepare_args(args, DEFAULT_ADD_MS2LDA_CONFIG)
    add_ms2lda_config = AddMS2LDAConfig(**dict(vars(args)))
    add_ms2lda_config.display()

    add_ms2lda(add_ms2lda_config)


def add_ms2lda(params: AddMS2LDAConfig) -> None:
    import numpy as np
    import tomotopy as tp
    from matchms import Spectrum

    from src.utils.motifs import run_overlap_scores_calculation

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

    result = run_overlap_scores_calculation(spectra, model)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--defaults", type=str, default=None, help="Path to YAML config file."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to trained LDA model file (.lda).",
    )
    parser.add_argument(
        "--graph",
        "-g",
        required=True,
        type=str,
        help="Path to target network CX graph file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Motif metadata overlap threshold.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
