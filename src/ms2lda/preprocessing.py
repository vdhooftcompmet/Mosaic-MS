from typing import Generator
import matchms.filtering as msfilters
from matchms import Spectrum
from utils.configs import MS2LDAConfig


def clean_spectra(spectra: Generator[Spectrum, None, None], config: MS2LDAConfig) -> list[Spectrum]:
    result = []

    spectra = [x for x in spectra]

    for spectrum in spectra:
        spectrum = msfilters.default_filters(spectrum)
        spectrum = msfilters.add_retention_index(spectrum)
        spectrum = msfilters.add_retention_time(spectrum)
        spectrum = msfilters.normalize_intensities(spectrum)
        spectrum = msfilters.select_by_relative_intensity(spectrum, intensity_from=config.prep_min_intensity, intensity_to=config.prep_max_intensity)
        spectrum = msfilters.select_by_mz(spectrum, mz_from=config.prep_min_mz, mz_to=config.prep_max_mz)
        spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=config.prep_max_frags)
        spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=config.prep_min_frags)

        if spectrum:
            result.append(spectrum)

    for i, spectrum in enumerate(result):
        spectrum.set("spectrum_id", i)

    return result


def spectra_to_documents(spectra: list[Spectrum], config: MS2LDAConfig) -> list[list[str]]:

    assert config.dataset_acquisition_type in ["DDA", "DIA"]

    result = []

    for spectrum in spectra:

        spectrum = msfilters.normalize_intensities(spectrum)
        
        document: list[str] = []
        result.append(document)

        for mz, intensity in zip(spectrum.peaks.mz, spectrum.peaks.intensities):
            rounded_intensity = int((intensity * 100).round())
            rounded_ms_value = round(mz, config.dataset_significant_digits)
            word = f"frag@{rounded_ms_value}"

            for _ in range(rounded_intensity):
                document.append(word)

        if config.dataset_acquisition_type == "DIA":
            continue

        for mz, intensity in zip(spectrum.losses.mz, spectrum.peaks.intensities):
            rounded_intensity = int((intensity * 100).round())
            rounded_ms_value = round(mz, config.dataset_significant_digits)
            word = f"loss@{rounded_ms_value}"

            for _ in range(rounded_intensity):
                if rounded_ms_value <= 0.01:
                    continue
                document.append(word)

    return result
