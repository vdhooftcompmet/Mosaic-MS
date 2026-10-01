import argparse
import json
import os
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rich.filesize import decimal
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# Disable RDKit C++ warnings and errors (e.g., ambiguous stereochemistry, sanitization/valence failures)
rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")

REPO = Path(__file__).parent.parent.parent.resolve()
DOWNLOAD_LINK = (
    "https://zenodo.org/records/13692394/files/coconut-09-2024.sdf.zip?download=1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download an SDF file (or ZIP containing an SDF) and convert it into a JSONL structure database."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DOWNLOAD_LINK,
        help="SDF file download URL",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=str(REPO / "data"),
        help="Directory where the downloaded file will be saved (default: ./data)",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=str(REPO / "data" / "coconut_db.jsonl"),
        help="Path to the output JSONL file (default: output.jsonl)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    downloaded_path = download_from_link(args.url, args.target_dir)

    # Extract if it is a zip file
    sdf_path = handle_extraction_if_zip(downloaded_path, args.target_dir)

    sdf_to_structure_db(sdf_path, args.output_jsonl)


def handle_extraction_if_zip(filepath: str | Path, target_dir: str | Path) -> str:
    """Extracts zip files if necessary and returns the path to the .sdf file."""
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".zip" or zipfile.is_zipfile(filepath):
        print(f"Extracting zip file: {filepath.name}...")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(target_dir)
            extracted_files = zip_ref.namelist()

        # Find the .sdf file among extracted contents
        sdf_files = [
            f
            for f in extracted_files
            if f.endswith(".sdf") and not f.startswith("__MACOSX")
        ]

        if sdf_files:
            extracted_sdf = Path(target_dir) / sdf_files[0]
            print(f"Extracted SDF file to: {extracted_sdf}")
            return str(extracted_sdf)
        else:
            raise FileNotFoundError(
                "No .sdf file found inside the extracted zip archive."
            )

    return str(filepath)


def download_from_link(url: str, target_dir: str | Path) -> str:
    parsed_url = urlparse(url)
    raw_filename = os.path.basename(parsed_url.path)

    if not raw_filename:
        raw_filename = "downloaded_file"

    os.makedirs(str(target_dir), exist_ok=True)
    local_filename = os.path.join(target_dir, raw_filename)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    formatted_size = decimal(total_size)

    print(f"Downloading {url} ({formatted_size})")
    progress_bar = Progress(
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    with progress_bar as progress:
        task = progress.add_task("downloading...", total=total_size)

        with open(local_filename, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
                progress.update(task, advance=len(data))

    return str(local_filename)


def sdf_to_structure_db(input_sdf: str | Path, output_jsonl: str | Path) -> None:
    suppl = Chem.SDMolSupplier(str(input_sdf))
    total_mols = len(suppl)  # Read total length for exact progress bar
    seen = set()

    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
    )

    with progress_bar as progress, open(str(output_jsonl), "w", encoding="utf-8") as f:
        task = progress.add_task("Processing molecules...", total=total_mols)

        for mol in suppl:
            progress.update(task, advance=1)

            if not mol:
                continue

            Chem.RemoveStereochemistry(mol)
            inchi_key = Chem.MolToInchiKey(mol)

            if inchi_key in seen:
                continue

            seen.add(inchi_key)

            # Precompute Morgan Fingerprint (radius=2, 2048 bits)
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

            record = {
                "smiles": Chem.MolToSmiles(mol),
                "neutral_mass": Descriptors.ExactMolWt(mol),
                "inchikey": inchi_key,
                "morgan_fingerprint": fp.ToBase64(),
            }
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
