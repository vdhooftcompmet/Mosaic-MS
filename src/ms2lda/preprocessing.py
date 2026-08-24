import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from typing import Generator
import matchms.filtering as msfilters
from matchms import Spectrum
from argparse import Namespace


def clean_spectra(spectra: Generator[Spectrum, None, None], params: Namespace) -> list[Spectrum]:
    result = []

    spectra = [x for x in spectra]

    for spectrum in spectra:
        spectrum = msfilters.default_filters(spectrum)
        spectrum = msfilters.add_retention_index(spectrum)
        spectrum = msfilters.add_retention_time(spectrum)
        spectrum = msfilters.normalize_intensities(spectrum)
        spectrum = msfilters.select_by_relative_intensity(spectrum, intensity_from=params.prep_min_intensity, intensity_to=params.prep_max_intensity)
        spectrum = msfilters.select_by_mz(spectrum, mz_from=params.prep_min_mz, mz_to=params.prep_max_mz)
        spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=params.prep_max_frags)
        spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=params.prep_min_frags)

        if spectrum:
            result.append(spectrum)

    for i, spectrum in enumerate(result):
        spectrum.set("spectrum_id", i)

    return result


def spectra_to_documents(spectra: list[Spectrum], params: Namespace) -> list[list[str]]:

    assert params.dataset_acquisition_type in ["DDA", "DIA"]

    result = []

    for spectrum in spectra:

        spectrum = msfilters.normalize_intensities(spectrum)
        
        document: list[str] = []
        result.append(document)

        for mz, intensity in zip(spectrum.peaks.mz, spectrum.peaks.intensities):
            rounded_intensity = int((intensity * 100).round())
            rounded_ms_value = round(mz, params.dataset_significant_digits)
            word = f"frag@{rounded_ms_value}"

            for _ in range(rounded_intensity):
                document.append(word)

        if params.dataset_acquisition_type == "DIA": continue

        for mz, intensity in zip(spectrum.losses.mz, spectrum.peaks.intensities):
            rounded_intensity = int((intensity * 100).round())
            rounded_ms_value = round(mz, params.dataset_significant_digits)
            word = f"loss@{rounded_ms_value}"

            for _ in range(rounded_intensity):
                if rounded_ms_value <= 0.01:
                    continue
                document.append(word)

    return result