import argparse
import functools
import json
import logging
from pathlib import Path
from typing import Generator, Set

from rdkit import Chem
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

# Suppress matchms logging warnings
logging.getLogger("matchms").setLevel(logging.ERROR)


def _import_matchms_io():
    """Lazy import for matchms I/O functions."""
    from matchms.exporting import save_as_mgf
    from matchms.importing import load_from_mgf

    logging.getLogger("matchms").setLevel(logging.ERROR)
    return load_from_mgf, save_as_mgf


def get_progress_bar() -> Progress:
    """Instantiates a rich progress bar layout matching the custom configuration."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
    )


def count_lines(file_path: Path) -> int:
    """Counts total lines in a file for accurate rich progress tracking."""
    count = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            count += 1
    return count


def count_mgf_spectra(input_path: Path) -> int:
    """Counts spectra by scanning 'BEGIN IONS' tags without loading full objects into memory."""
    count = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().upper().startswith("BEGIN IONS"):
                count += 1
    return count


def safe_load_from_mgf(mgf_path: Path) -> Generator:
    """Wraps matchms load_from_mgf generator to handle corrupted/malformed MGF blocks gracefully."""
    load_from_mgf, _ = _import_matchms_io()
    mgf_gen = load_from_mgf(str(mgf_path))

    while True:
        try:
            spectrum = next(mgf_gen)
            if spectrum is not None:
                yield spectrum
        except StopIteration:
            break
        except Exception as e:
            print(f"\nWarning: Skipped malformed spectrum entry in MGF: {e}")
            continue


@functools.lru_cache(maxsize=100000)
def smiles_to_inchikey(smiles: str) -> str | None:
    """Converts SMILES to InChIKey via RDKit with stereochemistry removed."""
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    Chem.RemoveStereochemistry(mol)
    return Chem.MolToInchiKey(mol)


@functools.lru_cache(maxsize=100000)
def inchi_to_inchikey(inchi: str) -> str | None:
    """Converts InChI string to InChIKey via RDKit with stereochemistry removed."""
    if not inchi:
        return None

    mol = Chem.MolFromInchi(inchi)
    if not mol:
        return None

    Chem.RemoveStereochemistry(mol)
    return Chem.MolToInchiKey(mol)


def derive_inchikey_rdkit(structure_dict: dict) -> str | None:
    """Derives standard InChIKey from dictionary metadata using RDKit."""
    # 1. Existing InChIKey field
    ik = (
        structure_dict.get("inchikey")
        or structure_dict.get("inchi_key")
        or structure_dict.get("InChIKey")
    )
    if ik and isinstance(ik, str) and len(ik.strip()) >= 14:
        return ik.strip()

    # 2. Derive from SMILES
    smiles = structure_dict.get("smiles") or structure_dict.get("SMILES")
    if smiles:
        ik = smiles_to_inchikey(str(smiles))
        if ik:
            return ik

    # 3. Derive from InChI
    inchi = structure_dict.get("inchi") or structure_dict.get("InChI")
    if inchi:
        ik = inchi_to_inchikey(str(inchi))
        if ik:
            return ik

    return None


def read_structural_db_inchikeys(
    jsonl_path: str | Path, progress: Progress
) -> Set[str]:
    """Reads JSONL database and extracts InChIKeys using rich progress visual updates."""
    jsonl_path = Path(jsonl_path)
    total_lines = count_lines(jsonl_path)
    inchikeys = set()

    task = progress.add_task("Processing JSONL database...", total=total_lines)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line {i} in {jsonl_path}")
                progress.update(task, advance=1)
                continue

            inchikey = derive_inchikey_rdkit(record)
            if inchikey:
                inchikeys.add(inchikey)

            progress.update(task, advance=1)

    return inchikeys


def get_or_derive_spectrum_inchikey(spectrum) -> str | None:
    """Derives InChIKey for matchms Spectrum object strictly via RDKit."""
    existing_ik = spectrum.get("inchikey") or spectrum.get("inchi_key")
    if existing_ik:
        return existing_ik

    derived_ik = derive_inchikey_rdkit(spectrum.metadata)
    if derived_ik:
        spectrum.set("inchikey", derived_ik)

    return derived_ik


def filter_mgf_by_database(
    spectral_database_mgf: Path,
    structure_database_jsonl: Path,
    output_mgf: Path,
) -> None:
    """Filters MGF spectra by intersecting with JSONL database InChIKeys without altering the input MGF."""
    # Lazy-load matchms saving function
    _, save_as_mgf = _import_matchms_io()

    with get_progress_bar() as progress:
        # 1. Parse JSONL keys
        db_inchikeys = read_structural_db_inchikeys(structure_database_jsonl, progress)

        # 2. Count MGF spectra to define progress length
        total_spectra = count_mgf_spectra(spectral_database_mgf)

        matched_spectra = []

        task = progress.add_task("Filtering MGF spectra...", total=total_spectra)

        for spectrum in safe_load_from_mgf(spectral_database_mgf):
            inchikey = get_or_derive_spectrum_inchikey(spectrum)

            if inchikey and inchikey in db_inchikeys:
                matched_spectra.append(spectrum)

            progress.update(task, advance=1)

    print("=" * 60)
    print(f"  • JSONL database unique InChIKeys: {len(db_inchikeys)}")
    print(f"  • Total matched spectra retained:  {len(matched_spectra)}")

    output_mgf.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(matched_spectra)} matching spectra to {output_mgf}...")
    save_as_mgf(matched_spectra, str(output_mgf), file_mode="w")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Filter MGF spectra by keeping only those with InChIKeys found in a JSONL structure database."
    )
    parser.add_argument(
        "-j",
        "--jsonl",
        type=Path,
        required=True,
        help="Path to input JSONL structure database file",
    )
    parser.add_argument(
        "-i",
        "--input-mgf",
        type=Path,
        required=True,
        help="Path to input MGF spectral file",
    )
    parser.add_argument(
        "-o",
        "--output-mgf",
        type=Path,
        default=None,
        help="Path for output filtered MGF file (default: input_filename_intersection.mgf)",
    )

    args = parser.parse_args()

    jsonl_path = args.jsonl.resolve()
    if not jsonl_path.exists():
        parser.error(f"JSONL file not found: {jsonl_path}")

    input_mgf_path = args.input_mgf.resolve()
    if not input_mgf_path.exists():
        parser.error(f"Input MGF file not found: {input_mgf_path}")

    output_mgf_path = args.output_mgf
    if output_mgf_path is None:
        output_mgf_path = (
            input_mgf_path.parent / f"{input_mgf_path.stem}_intersection.mgf"
        )
    else:
        output_mgf_path = output_mgf_path.resolve()

    print("=" * 60)
    print("MGF Database Intersection Configured Options:")
    print(f"  • JSONL Input Path:   {jsonl_path}")
    print(f"  • MGF Input Path:     {input_mgf_path}")
    print(f"  • MGF Output Path:    {output_mgf_path}")
    print("=" * 60)

    filter_mgf_by_database(
        spectral_database_mgf=input_mgf_path,
        structure_database_jsonl=jsonl_path,
        output_mgf=output_mgf_path,
    )


if __name__ == "__main__":
    main()
