import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from typing import Generator
from pathlib import Path

from matchms import Spectrum
from matchms.importing import load_from_mgf
from matchms.filtering.default_pipelines import DEFAULT_FILTERS, CLEAN_PEAKS
from matchms.filtering.SpectrumProcessor import SpectrumProcessor

from matchms.exporting import save_as_mgf
from utils.constants import *


def clean_mgf(path: Path | str):
    assert isinstance(path, str) or isinstance(path, Path), "path must be a Path object or a string"

    spectra = load_from_mgf(path)
    spectrum_processor = SpectrumProcessor(DEFAULT_FILTERS + CLEAN_PEAKS)
    result, _ = spectrum_processor.process_spectra(spectra, progress_bar=False)

    for i, spectrum in enumerate(result):
        spectrum.set(KEY_SPECTRUM_ID, i)

    return result

def main(params):
    cleaned_spectra = list(clean_mgf(params.raw_mgf))
    save_as_mgf(cleaned_spectra, params.cleaned_mgf, file_mode="w")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--raw-mgf", type=str, required=True)
    parser.add_argument("--cleaned-mgf", type=str, required=True)
    params = parser.parse_args()
    main(params)
