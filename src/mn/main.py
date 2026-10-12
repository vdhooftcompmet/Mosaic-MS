import argparse
import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from matchms import Spectrum
from scipy.sparse import csr_matrix, load_npz, save_npz

from src.utils.cli import prepare_args
from src.utils.configs import MNConfig
from src.utils.cx import write_cx
from src.utils.paths import MN_STYLE_FILE

REPO = Path(__file__).parent.parent.parent.resolve()
DEFAULT_MN_CONFIG = REPO / "config" / "mn.yaml"


def main(args: argparse.Namespace) -> None:
    args = prepare_args(args, DEFAULT_MN_CONFIG)
    config = MNConfig(**dict(vars(args)))
    config.display()

    from src.mn.mn import (
        add_cluster_numbering,
        calculate_bootstrapping,
        clean_mgf,
        run_networking,
    )

    spectra = clean_mgf(config.mgf)
    similarity, support = _similarity_cache(calculate_bootstrapping)(spectra, config)

    file_names = {
        "base": config.base_graph_path,
        "threshold": config.threshold_graph_path,
        "rescued": config.rescued_graph_path,
    }

    for graph_type, file_name in file_names.items():
        graph = run_networking(spectra, similarity, support, graph_type, config)
        add_cluster_numbering(graph)

        write_cx(graph, file_name, MN_STYLE_FILE)


def _similarity_cache(
    fn: Callable,
) -> Callable[[list[Spectrum], MNConfig], tuple[np.ndarray, np.ndarray]]:

    from src.mn.mn import plain_similarity

    def inner(
        spectra: list[Spectrum], config: MNConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        files = {}
        for file_type in ["avg_sim", "tot_sim", "tot_sup"]:
            files[file_type] = Path(config.cache_folder) / _make_cache_name(
                file_type, config
            )

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
    hasher.update(str(file_type).lower().encode("utf-8"))
    hasher.update(str(config.k).lower().encode("utf-8"))
    hasher.update(str(config.B).lower().encode("utf-8"))
    hasher.update(str(config.seed).lower().encode("utf-8"))
    hexadecimal_string = hasher.hexdigest()
    file_name = f"{config.similarity_type}-{hexadecimal_string[:20]}.npz"
    return file_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI tool for strata-ms workflows.")
    parser.add_argument(
        "--defaults",
        type=str,
        default=None,
        help="Path to a YAML configuration file containing default parameters.",
    )
    parser.add_argument(
        "--mgf",
        "-m",
        type=str,
        default=None,
        help="File path pointing towards the mass spectra (MGF) to be analyzed.",
    )
    parser.add_argument(
        "--similarity-type",
        choices=["cos", "modcos", "spec2vec", "ms2deepscore"],
        default=None,
        help="Similarity metric used for networking.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Minimum similarity threshold required to connect two nodes with an edge.",
    )
    parser.add_argument(
        "--flash-tolerance",
        type=float,
        default=None,
        help="Mass tolerance error margin (in Da).",
    )
    parser.add_argument(
        "--binning-decimals",
        type=int,
        default=None,
        help="Precision used to bin spectrum m/z values.",
    )
    parser.add_argument(
        "--max-component-size",
        type=int,
        default=None,
        help="Maximum allowable cluster size in the network.",
    )
    parser.add_argument(
        "--B", type=int, default=None, help="Bootstrapping iterations per spectrum."
    )
    parser.add_argument("--k", type=int, default=None, help="Top-k nearest neighbors.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--ms2deepscore-model-path",
        type=str,
        default=None,
        help="MS2DeepScore model path.",
    )
    parser.add_argument(
        "--spec2vec-model-path", type=str, default=None, help="Spec2Vec model path."
    )
    parser.add_argument(
        "--use-average-similarity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use average similarity from bootstrapping.",
    )
    parser.add_argument(
        "--cache-similarity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache and reuse similarity calculations.",
    )
    parser.add_argument(
        "--base-graph-path",
        type=str,
        default=None,
        help="Path for base network graph (.cx).",
    )
    parser.add_argument(
        "--threshold-graph-path",
        type=str,
        default=None,
        help="Path for threshold graph (.cx).",
    )
    parser.add_argument(
        "--rescued-graph-path",
        type=str,
        default=None,
        help="Path for rescued graph (.cx).",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
