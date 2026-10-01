import argparse
import logging
from pathlib import Path
from typing import Any, Generator, Optional, Set, Tuple

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

# Suppress matchms logging warnings in the terminal
logging.getLogger("matchms").setLevel(logging.ERROR)


def _import_matchms_io():
    """Lazy import for matchms I/O functions."""
    from matchms.exporting import save_as_mgf
    from matchms.importing import load_from_mgf
    logging.getLogger("matchms").setLevel(logging.ERROR)

    return load_from_mgf, save_as_mgf


def _import_matchms_processor():
    """Lazy import for matchms SpectrumProcessor and standard pipelines."""
    from matchms.filtering.default_pipelines import CLEAN_PEAKS, LIBRARY_CLEANING
    from matchms.filtering.SpectrumProcessor import SpectrumProcessor

    return SpectrumProcessor, LIBRARY_CLEANING, CLEAN_PEAKS


def get_progress_bar() -> Progress:
    """Helper to instantiate a standard rich progress bar layout."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
    )


def count_mgf_spectra(input_path: Path) -> int:
    """Count total spectra by scanning 'BEGIN IONS' tags in the file without loading full spectra."""
    count = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().upper().startswith("BEGIN IONS"):
                count += 1
    return count


def print_active_options(
    args: argparse.Namespace, input_path: Path, output_path: Path
) -> None:
    """Display selected configuration options to the user in the CLI."""
    print("=" * 60)
    print("MGF Streaming & Filtering Pipeline Configured Options:")
    print(f"  • Input Path:         {input_path}")
    print(f"  • Output Path:        {output_path}")
    print(
        f"  • Matchms Cleaning:   {'Enabled (LIBRARY_CLEANING + CLEAN_PEAKS)' if args.clean else 'Disabled'}"
    )
    print(
        f"  • Unique InChIKey:    {'Enabled' if args.unique_inchikey else 'Disabled'}"
    )

    if args.keep_metadata:
        rules_str = "; ".join([f"{k} in {v}" for k, v in args.keep_metadata])
        print(f"  • Keep Metadata:      {rules_str}")
    else:
        print("  • Keep Metadata:      None")

    if args.discard_metadata:
        rules_str = "; ".join([f"{k} in {v}" for k, v in args.discard_metadata])
        print(f"  • Discard Metadata:   {rules_str}")
    else:
        print("  • Discard Metadata:   None")
    print("=" * 60)


def clean_single_spectrum(spectrum: Any, processor: Any) -> Optional[Any]:
    """Clean a single spectrum using matchms SpectrumProcessor without stdout progress output."""
    result, _ = processor.process_spectra(
        [spectrum],
        create_report=False,
        progress_bar=False,
    )
    if result and result[0] is not None:
        return result[0]
    return None


def filter_single_by_metadata(
    spectrum: Any,
    keep_rules: list[Tuple[str, list[str]]],
    discard_rules: list[Tuple[str, list[str]]],
) -> Optional[Any]:
    """Evaluate metadata rules on a single spectrum. Returns the spectrum if kept, or None if discarded."""
    # Check positive retention rules
    for key, allowed_vals in keep_rules:
        val = str(spectrum.get(key, "")).strip().lower()
        allowed_vals_clean = [v.strip().lower() for v in allowed_vals]
        if val not in allowed_vals_clean:
            return None

    # Check negative exclusion rules
    for key, discard_vals in discard_rules:
        val = str(spectrum.get(key, "")).strip().lower()
        discard_vals_clean = [v.strip().lower() for v in discard_vals]
        if val in discard_vals_clean:
            return None

    return spectrum


def filter_single_unique_inchikey(spectrum: Any, seen_keys: Set[str]) -> Optional[Any]:
    """Check single spectrum against seen InChIKeys set. Updates set in-place if key is unique."""
    ik = (
        spectrum.get("inchikey")
        or spectrum.get("inchi_key")
        or spectrum.get("InChIKey")
    )

    if not ik and spectrum.get("smiles"):
        ik = spectrum.get("smiles")

    if not ik:
        return spectrum

    ik_clean = str(ik).strip()
    ik_skeleton = ik_clean.split("-")[0]

    if ik_clean in seen_keys or ik_skeleton in seen_keys:
        return None

    seen_keys.add(ik_clean)
    seen_keys.add(ik_skeleton)
    return spectrum


def parse_metadata_rule(rule_str: str) -> Tuple[str, list[str]]:
    """Parse 'key=val1,val2' into ('key', ['val1', 'val2'])."""
    if "=" not in rule_str:
        raise argparse.ArgumentTypeError(
            f"Invalid metadata rule format '{rule_str}'. Expected 'key=value1,value2'."
        )
    key, val_string = rule_str.split("=", 1)
    values = [v.strip() for v in val_string.split(",") if v.strip()]
    return key.strip(), values


def stream_mgf_spectra(input_path: Path) -> Generator[Any, None, None]:
    """Stream spectra one-by-one directly from matchms MGF reader generator."""
    load_from_mgf, _ = _import_matchms_io()
    for spectrum in load_from_mgf(str(input_path)):
        if spectrum is not None:
            yield spectrum


def main():
    parser = argparse.ArgumentParser(
        description="Stream, clean, and filter MGF mass spectrometry files line-by-line."
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True, help="Path to input .mgf file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path for output filtered .mgf file",
    )

    # Cleaning flag defaulting to True with a --no-clean override
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply matchms SpectrumProcessor cleaning (default: True, disable with --no-clean)",
    )

    # Unique InChIKey flag defaulting to True with a --no-unique-inchikey override
    parser.add_argument(
        "--unique-inchikey",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter spectra to keep unique InChIKeys (default: True, disable with --no-unique-inchikey)",
    )

    parser.add_argument(
        "--keep-metadata",
        action="append",
        type=parse_metadata_rule,
        default=[],
        help="Filter metadata rule to keep matching spectra: --keep-metadata 'ionmode=positive,pos'",
    )
    parser.add_argument(
        "--discard-metadata",
        action="append",
        type=parse_metadata_rule,
        default=[],
        help="Filter metadata rule to discard matching spectra: --discard-metadata 'ms_level=MS1'",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    output_path = args.output
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_filtered.mgf"
    else:
        output_path = output_path.resolve()

    print_active_options(args, input_path, output_path)

    # Fast file scan to get accurate spectrum count for progress bar
    print(f"Counting spectra in {input_path.name}...")
    total_spectra = count_mgf_spectra(input_path)
    print(f"Found {total_spectra} total spectra.")

    # Lazy-load matchms saving function
    _, save_as_mgf = _import_matchms_io()

    # Lazy-load and initialize processor if requested
    processor = None
    if args.clean:
        SpectrumProcessor, LIBRARY_CLEANING, CLEAN_PEAKS = _import_matchms_processor()
        processor = SpectrumProcessor(LIBRARY_CLEANING + CLEAN_PEAKS)

    seen_inchikeys: Set[str] = set()

    # Clear/create target destination file before appending streaming results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_read = 0
    total_retained = 0

    progress_bar = get_progress_bar()
    with progress_bar as progress:
        task = progress.add_task(
            "Streaming and processing spectra...", total=total_spectra
        )

        for spectrum in stream_mgf_spectra(input_path):
            total_read += 1

            # 1. Matchms spectrum cleaning
            if processor:
                spectrum = clean_single_spectrum(spectrum, processor)
                if spectrum is None:
                    progress.update(task, completed=total_read)
                    continue

            # 2. Metadata filtering rules
            if args.keep_metadata or args.discard_metadata:
                spectrum = filter_single_by_metadata(
                    spectrum, args.keep_metadata, args.discard_metadata
                )
                if spectrum is None:
                    progress.update(task, completed=total_read)
                    continue

            # 3. Unique InChIKey deduplication
            if args.unique_inchikey:
                spectrum = filter_single_unique_inchikey(spectrum, seen_inchikeys)
                if spectrum is None:
                    progress.update(task, completed=total_read)
                    continue

            # 4. Immediate write-to-file in append mode
            save_as_mgf([spectrum], str(output_path), file_mode="a")
            total_retained += 1

            progress.update(task, completed=total_read)

    print("=" * 60)
    print("Finished streaming pipeline.")
    print(f"  • Total spectra read:     {total_read}")
    print(f"  • Total spectra retained: {total_retained}")
    print(f"  • Saved output to:        {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
