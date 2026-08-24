import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.autolayout'] = False

import matplotlib.patheffects as pe

from matplotlib.axes import Axes
from typing import Any
from matchms import Spectrum, Fragments


def comparison_plot(references: list | Spectrum | None = None, spectra: list | Spectrum | None = None, scale: float = 1.6, threshold=0.005, top_n=35) -> list[Axes]:
    if references is None:
        references = []

    elif isinstance(references, Spectrum):
        references = [references]

    if spectra is None:
        spectra = []

    elif isinstance(spectra, Spectrum):
        spectra = [spectra]

    references = [add_losses(r) for r in references]
    spectra    = [add_losses(s) for s in spectra]

    peak_values = [s.peaks.mz for s in spectra + references]
    peak_values = np.concatenate(peak_values)

    if len(peak_values) == 0:
        print("WARNING, no features detected in any spectra")
        fig, ax = plt.subplots()
        return [ax]

    min_mz, max_mz = min(peak_values), max(peak_values)
    min_mz = min(min_mz, 0)

    mz_range = max_mz - min_mz
    min_mz = min_mz - 0.05 * mz_range
    max_mz = max_mz + 0.05 * mz_range

    colors = ["black"] + plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]

    reference_axes = plot_references(references, colors, min_mz, max_mz, scale, threshold, top_n)
    spectra_axes = plot_spectra(spectra, references, colors, min_mz, max_mz, scale, threshold, top_n)

    return reference_axes + spectra_axes


def plot_references(references: list[Spectrum], colors: Any, min_mz: float, max_mz: float, scale: float, threshold, top_n) -> Axes:
    axes = []

    for spectrum, color in zip(references, colors):

        mz = np.round(spectrum.peaks.mz, 2)
        peak_colors = np.array([color for _ in range(mz.size)])
        ax = create_plot_grid(min_mz, max_mz, scale)
        plot_features(ax, mz, spectrum.peaks.intensities, peak_colors=peak_colors, title=str(spectrum.get("id")), threshold=threshold, top_n=top_n)

        axes.append(ax)

    return axes


def plot_spectra(spectra: list[Spectrum], references: list[Spectrum], colors: Any, min_mz: float, max_mz: float, scale: float, threshold, top_n) -> Axes:
    axes = []

    for spectrum in spectra:
        mz = spectrum.peaks.mz
        intensities = spectrum.peaks.intensities
        
        peak_colors = np.array(["black" for _ in mz], dtype=object)

        for reference, color in zip(references, colors):
            ref_mz = np.round(reference.peaks.mz, 2)

            matching_grid = np.isclose(mz[:, None], ref_mz[None, :], atol=0.01)
            intersection = matching_grid.any(axis=1)

            peak_colors[intersection] = color

        ax = create_plot_grid(min_mz, max_mz, scale)
        plot_features(ax, mz, intensities, peak_colors=peak_colors, title="spectrum " + str(spectrum.get("spectrum_id")), threshold=threshold, top_n=top_n)

        axes.append(ax)

    return axes


def add_losses(spectrum: Spectrum) -> Spectrum:
    precursor_mz = spectrum.get("precursor_mz")

    if np.any(spectrum.peaks.mz < 0):
        return spectrum

    if precursor_mz is None:
        return spectrum
    
    # remove peaks larger than precursor mz 
    valid_peak_indices = spectrum.peaks.mz <= precursor_mz

    peaks_mz = spectrum.peaks.mz[valid_peak_indices]
    peaks_intensities = spectrum.peaks.intensities[valid_peak_indices]
    
    # compute losses
    losses_mz = peaks_mz - precursor_mz
    losses_intensities = peaks_intensities

    # create new paeks and losses
    mz = np.concatenate([losses_mz, peaks_mz])
    intensities = np.concatenate([losses_intensities, peaks_intensities])

    peaks = Fragments(mz=mz, intensities=intensities)
    spectrum.peaks = peaks

    return spectrum
    

def plot_features(ax, mz, intensities, peak_colors, title="", threshold=0.005, top_n=35) -> None:
    ax.vlines(mz, ymin=0, ymax=intensities, colors=peak_colors, linewidth=1.0, zorder=5)
    ax.set_title(title)

    valid_mzs, valid_intensities, valid_peak_colors = filter_labels(mz, intensities, peak_colors, threshold, top_n)

    x_min, x_max = ax.get_xlim()
    if x_min == 0.0 and x_max == 1.0: 
        x_min, x_max = np.min(mz), np.max(mz)
    
    min_gap = (x_max - x_min) * 0.015  

    adjusted_mzs = valid_mzs.copy()
    for i in range(1, len(adjusted_mzs)):
        if adjusted_mzs[i] - adjusted_mzs[i-1] >= min_gap:
            continue
        adjusted_mzs[i] = adjusted_mzs[i-1] + min_gap

    path_effects = [pe.withStroke(linewidth=2, foreground="white")]

    for original_mz, intensity, shifted_mz, peak_color in zip(valid_mzs, valid_intensities, adjusted_mzs, valid_peak_colors):
        arrow_props = dict(
            arrowstyle="-",
            color=peak_color,
            lw=0.5,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0"
        )
        ax.annotate(
            "", 
            xy=(original_mz, 0.0),      
            xytext=(shifted_mz, -12),    
            xycoords="data",
            textcoords=("data", "offset points"),
            arrowprops=arrow_props,
            annotation_clip=False
        ) 
        ax.annotate(
            f"{original_mz:.4f} : {intensity:.2f}",
            xy=(shifted_mz, 0.0),                  
            xytext=(0, -14),                       
            xycoords="data",                       
            textcoords="offset points",            
            rotation=90,
            fontsize=5,
            color=peak_color,
            ha="center", 
            va="top",    
            path_effects=path_effects,
            annotation_clip=False
        )


def top_indexes(arr, threshold, top_n):
    valid_mask = arr > threshold
    valid_indices = np.where(valid_mask)[0]
    valid_values = arr[valid_indices]

    if len(valid_values) <= top_n:
        return valid_indices[np.argsort(valid_values)[::-1]]
    else:
        top_subset_sorted_idx = np.argsort(valid_values)[-top_n:][::-1]
        return valid_indices[top_subset_sorted_idx]



def filter_labels(mz, intensities, peak_colors, threshold, top_n):
    mask = top_indexes(intensities, threshold=threshold, top_n=top_n)

    valid_intensities = intensities[mask]
    valid_mzs = mz[mask]
    valid_colors = peak_colors[mask]

    if valid_mzs.size == 0:
        return np.array([]), np.array([]), np.array([])

    sort_idx = np.argsort(valid_mzs)
    valid_mzs = valid_mzs[sort_idx]
    valid_intensities = valid_intensities[sort_idx]
    valid_colors = valid_colors[sort_idx]

    return valid_mzs, valid_intensities, valid_colors


def create_plot_grid(min_mz, max_mz, scale=1):
    fig, ax = plt.subplots(figsize=(4 * scale, 1 * scale), dpi=300)  # inches
    plt.close(fig)

    ax.set_xlim(min_mz, max_mz)
    ax.set_ylim(0, 1.1)

    ax.grid(visible=True, which="major", color="#9E9E9E", linewidth=0.2)
    ax.grid(visible=True, which="minor", color="#9E9E9E", linewidth=0.2)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="both", labelsize="small")
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks[y_ticks <= 1.0])

    ax.set_xlabel("m/z", style="italic", labelpad=36)
    ax.set_ylabel("Intensity")
    return ax
