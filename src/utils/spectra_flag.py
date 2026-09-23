

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--mn-file", required=True, type=str)
    parser.add_argument("--annotation-folder", required=True, type=str)
    parser.add_argument("--result-folder", required=True, type=str)
    parser.add_argument("--intersection", required=True, type=str)
    parser.add_argument("--mgf", required=True, type=str)
    parser.add_argument("--ms2deepscore-model-path", type=str)
    parser.add_argument("--spec2vec-model-path", type=str)
    parser.add_argument("--similarity-threshold", required=True, type=float)
    parser.add_argument("--similarity-type", required=True, type=str)
    
    params = parser.parse_args()


import numpy as np
from argparse import Namespace
from collections import defaultdict
from matchms.importing import load_from_mgf
from tqdm import tqdm
from pathlib import Path


from utils.cx import write_cx, read_cx, read_annotations
from setup.paths import ANNOTATION_STYLE_FILE, MN_STYLE_FILE

from utils.constants import *
from utils.plots import comparison_plot
from mn.bootstrap import get_similarity

from matchms import Spectrum
from matchms.importing import load_from_mgf
from matchms.similarity.FlashSimilarity import FlashSimilarity
from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore
from spec2vec import Spec2Vec
from unittest.mock import patch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from utils.folders import prepare_directory
from utils.constants import *
from utils.plots import *
from utils.images import *
from motif_metadata.motif_detection import run_overlap_scores_calculation


def main(params: Namespace) -> None:
    if not Path(params.result_folder).exists():
        prepare_directory(params.result_folder)

    mn = read_cx(params.mn_file)

    annotations, files, cluster_ids = read_annotations(params.annotation_folder)

    annotation_dict = dict(zip(cluster_ids, annotations))

    similarity_metric = get_similarity(params.similarity_type, 0.01, params.ms2deepscore_model_path, params.spec2vec_model_path)
    sample_lookup = build_sample_lookup(params)
    intersection_lookup = build_intersection_lookup(params)
    
    set_spectra_flag(mn, annotation_dict, similarity_metric, intersection_lookup, sample_lookup)

    write_cx(mn, params.mn_file, MN_STYLE_FILE)
    for file, annotation in zip(files, annotations):
        write_cx(annotation, file, ANNOTATION_STYLE_FILE)


def build_intersection_lookup(params):
    intersection = load_from_mgf(params.intersection)

    intersection_lookup = defaultdict(list)
    
    for spectrum in tqdm(intersection):
        key = str(spectrum.get(KEY_INCHI_KEY))
        if key is None:
            print(f"Warning: {KEY_INCHI_KEY} is None {spectrum = }")
        intersection_lookup[spectrum.get(KEY_INCHI_KEY)].append(spectrum)

    return intersection_lookup


def build_sample_lookup(params):
    original_spectra = load_from_mgf(params.mgf)

    sample_lookup = dict()
    for spectrum in tqdm(original_spectra):
        key = str(spectrum.get(KEY_SPECTRUM_ID))
        if key is None:
            print(f"Warning: spectrum_id is None {spectrum = }")
        sample_lookup[key] = spectrum

    return sample_lookup


def get_name(spectrum, lib_spectrum):
    spectrum_id = spectrum.get("spectrum_id")
    lib_id = lib_spectrum.get("spectrum_id")

    return  f"query_{spectrum_id}_{lib_id}_matches.svg"


def set_spectra_flag(mn, annotations, similarity_metric, intersection_lookup, sample_lookup):
    clusters = defaultdict(list)

    for node in mn:
        cluster_id = mn.nodes[node][KEY_MN_CLUSTER_ID]
        clusters[ str(cluster_id) ].append(node)

    for cluster_index, annotation in annotations.items():

        for node in annotation:
            annotation.nodes[node][KEY_HAS_SPECTRUM_MATCH] = "na"

        cluster = clusters.get( str(cluster_index) , [])

        for node in annotation:
            inchikey = annotation.nodes[node].get(KEY_INCHI_KEY)
            if inchikey not in intersection_lookup:
                continue

            annotation_spectra = intersection_lookup[inchikey]

            parents = annotation.nodes[node].get(KEY_MN_NODE_ID).split(";")

            parent_spectra = []

            for parent_node in cluster:
                if str(parent_node) not in parents:
                    continue

                parent_id = str( mn.nodes[parent_node].get(KEY_SPECTRUM_ID) )
                if parent_id not in sample_lookup:
                    continue

                parent_spectra += [ sample_lookup[parent_id] ]

            similarity = similarity_metric.matrix(annotation_spectra, parent_spectra)

            is_match = bool(np.max(similarity) >= float(params.similarity_threshold))
            annotation.nodes[node][KEY_HAS_SPECTRUM_MATCH] = "yes" if is_match else "no"

            x_dim, y_dim = similarity.shape
            for i in range(x_dim):
                for ii in range(y_dim):

                    sim = similarity[i, ii]
                    is_match = sim < params.similarity_threshold

                    ann = annotation_spectra[i].clone()
                    par = parent_spectra[ii].clone()
                    ann.set("score", sim)

                    svg = plot_spectra([par], [ann], ["precursor_mz", "ion", "adduct", "score"], 1.6) 

                    prefix = "MATCH" if is_match else ""

                    save_path = Path(params.result_folder) / f"{prefix}-{get_name(par, ann)}"
                    save_svg(svg, save_path)


def spectrum_metadata_to_text(spectrum: Spectrum, keys: list[str | int]) -> str:
    text_rows = []
    for k in keys:
        text_rows.append(f"{k}:")
        text_rows.append(f"{spectrum.get(k)}")

    text = "\n".join(text_rows)
    return text


def plot_spectra(reference: list[Spectrum], spectra: list[Spectrum], metadata_keys: list[str], scale: float):
    all_spectra = reference + spectra
    
    structure_svgs = [smiles_to_image(s.get("smiles"), scale) for s in all_spectra]
    
    axes = comparison_plot(reference, spectra, scale)
    ax_svgs = [ax_to_image(ax) for ax in axes]

    metadata_text = [spectrum_metadata_to_text(s, metadata_keys) for s in all_spectra]
    metadata_svgs = [text_to_image(t, scale) for t in metadata_text]

    columns = [
        list(ax_svgs) if not isinstance(ax_svgs, str) else [ax_svgs],
        list(structure_svgs) if not isinstance(structure_svgs, str) else [structure_svgs],
        list(metadata_svgs) if not isinstance(metadata_svgs, str) else [metadata_svgs]
    ]
    
    svg = make_vector_grid(columns)
    return svg


# TODO
if __name__ == "__main__":
    main(params)
    