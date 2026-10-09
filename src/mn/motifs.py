import argparse
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import tomotopy as tp
from matchms import Spectrum

from src.mn.mn import (
    add_cluster_numbering,
    clean_mgf,
)
from src.utils.cx import write_cx
from src.utils.motifs import run_overlap_scores_calculation
from src.utils.paths import MN_STYLE_FILE

logging.getLogger("matchms").setLevel(logging.ERROR)


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

        motif = topic_to_motif(model, motif_index)
        motif_metadata = motif.to_dict()
        motif_metadata = {f"motif_{k}": str(v) for k, v in motif_metadata.items()}

        for spectrum_index in spectrum_indices:
            spectrum = spectra[spectrum_index]
            metadata = {k: str(v) for k, v in spectrum.to_dict().items()}

            # !! add cluster numbering based on motif index, not based on size
            metadata |= {"mn_cluster_id": motif_index}

            # !! add motif peaks for later use
            metadata |= motif_metadata

            G.add_node(f"m{motif_index}s{spectrum_index}", **metadata)

        for u, spectrum_index_u in enumerate(spectrum_indices):
            for v, spectrum_index_v in enumerate(spectrum_indices):
                if u <= v:
                    continue

                id_u = f"m{motif_index}s{spectrum_index_u}"
                id_v = f"m{motif_index}s{spectrum_index_v}"
                G.add_edge(id_u, id_v)

    return G


def topic_to_motif(model: tp.LDAModel, topic_index: int) -> Spectrum:
    mz, intensities = [], []

    topic_words = model.get_topic_words(topic_index, 50)
    for word, probability in topic_words:
        mz_str = word[5:]
        mz.append(float(mz_str))

        if word.startswith("frag@"):
            intensities.append(probability)
        elif word.startswith("loss@"):
            intensities.append(probability * -1)
        else:
            raise ValueError()

    mz = np.array(mz)
    intensities = np.array(intensities)

    indices = np.argsort(mz)
    mz = mz[indices]
    intensities = intensities[indices]
    motif = Spectrum(np.array(mz), np.array(intensities), {"motif_id": topic_index})

    return motif


if __name__ == "__main__":
    main()
