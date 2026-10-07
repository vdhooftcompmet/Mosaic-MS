import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import tomotopy as tp

from src.mn.mn import (
    add_cluster_numbering,
    clean_mgf,
)
from src.utils.cx import write_cx
from src.utils.motifs import run_overlap_scores_calculation
from src.utils.paths import MN_STYLE_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m",
        "--mgf",
        required=True,
        type=Path,
        help="Path to mgf file",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="motif detection threshold",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=0.1,
        type=float,
        help="motif detection threshold",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="motifs.cx",
        help="Optional output graph .cx path (defaults to stdout)",
    )
    config = parser.parse_args()
    G = construct_motif_graph(config)
    write_cx(G, str(config.output), MN_STYLE_FILE)


def construct_motif_graph(params: argparse.Namespace) -> nx.Graph:
    model_path = Path(params.model).resolve()
    spectra = list(clean_mgf(str(params.mgf)))

    model = tp.LDAModel.load(str(model_path))
    result = run_overlap_scores_calculation(spectra, model)
    _beta_matrix, _phi_matrix, _theta_matrix, overlap_scores = result

    threshold = float(params.threshold)

    G = nx.Graph()

    for motif_index in range(overlap_scores.shape[0]):
        scores = overlap_scores[motif_index, :]

        assert len(scores) == len(spectra)

        spectrum_indices = np.where(scores > threshold)[0]

        for spectrum_index in spectrum_indices:
            spectrum = spectra[spectrum_index]
            metadata = {k: str(v) for k, v in spectrum.to_dict().items()}
            G.add_node(f"m{motif_index}s{spectrum_index}", **metadata)

        for u, spectrum_index_u in enumerate(spectrum_indices):
            for v, spectrum_index_v in enumerate(spectrum_indices):
                if u <= v:
                    continue

                id_u = f"m{motif_index}s{spectrum_index_u}"
                id_v = f"m{motif_index}s{spectrum_index_v}"
                G.add_edge(id_u, id_v)

    add_cluster_numbering(G)
    return G


if __name__ == "__main__":
    main()
