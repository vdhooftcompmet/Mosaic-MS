import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import argparse

from pathlib import Path

from utils.cli import print_params , add_defaults, completeness_check
from utils.cx import read_annotations


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mn-file", type=str)
    parser.add_argument("--overlaps", type=str)
    parser.add_argument("--threshold", type=float)
    return parser


if __name__ == "__main__":
    parser = cli()
    params = parser.parse_args()
    completeness_check(params)
    print("> running motif detection with parameters:\n")
    print_params(params)


import numpy as np

from pathlib import Path

from setup.paths import MN_STYLE_FILE
from utils.cx import write_cx, read_cx


def main(params):
    overlaps = np.loadtxt(params.overlaps, delimiter=",")
    threshold = float(params.threshold)

    mn = read_cx(params.mn_file)

    assert len(mn) == overlaps.shape[1], f"{len(mn)} vs {overlaps.shape[1]}"

    for node in mn:
        i = int(node)
        scores = overlaps[:, i]

        present_motifs = [str(m) for m, s in enumerate(scores) if s > threshold]
        mn.nodes[node]["motifs"] = ";".join(present_motifs)

        for ii, score in enumerate(scores):
            mn.nodes[node][f"motif_{ii}"] = 1 if score > threshold else 0

    write_cx(mn, params.mn_file, MN_STYLE_FILE)


if __name__ == "__main__":
    main(params)



