import argparse
from pathlib import Path

import networkx as nx
import numpy as np
from matchms import Spectrum

from src.mn.main import clean_mgf
from src.mn.mn import add_cluster_numbering, build_graph, filter_base_strategy
from src.utils.cx import write_cx
from src.utils.paths import MN_STYLE_FILE
from src.utils.similarity_matrix import similarity_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Compute annotation quality categories and output counts as a CSV."
    )
    parser.add_argument(
        "-m",
        "--mgf",
        required=True,
        type=Path,
        help="Path to mgf file",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=0.66,
        type=float,
        help="morgan dice similarity threshold",
    )
    parser.add_argument(
        "-s",
        "--max-component-size",
        default=100,
        type=int,
        help="max component size for graph clusters",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output graph .cx path (defaults to stdout)",
    )
    config = parser.parse_args()
    spectra = clean_mgf(str(config.mgf))
    G = build_ideal_graph(spectra, config)
    write_cx(G, str(config.output), MN_STYLE_FILE)


def build_ideal_graph(spectra: list[Spectrum], config) -> nx.Graph:
    smiles = [s.metadata["smiles"] for s in spectra]
    similarity = similarity_matrix(smiles, smiles, "morgan", "dice")
    support = np.ones_like(similarity)

    strat = filter_base_strategy(config.threshold)
    G = build_graph(spectra, similarity, support, strat, config.max_component_size)
    add_cluster_numbering(G)
    return G


if __name__ == "__main__":
    main()
