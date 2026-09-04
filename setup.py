import os
import requests
from pathlib import Path
import time
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from rich.filesize import decimal
from urllib.parse import urlparse


FILE_NAMES = [
    "coconut_db.jsonl",
    "input.mgf",
    "ms2deepscore_model.pt",
    "Spec2VecModel.model",
    "Spec2VecModel.model.syn1neg.npy",
    "Spec2VecModel.model.wv.vectors.npy"
]
WEB_ADDRESS = "https://zenodo.org/records/22230864/files/"
DOWNLOAD_SUFFIX = "?download=1"
DATA_DIR = Path("data")


def main():
    missing_files = []

    for file_name in FILE_NAMES:
        if (DATA_DIR / file_name).exists():
            continue

        missing_files.append(file_name)
        
    if not missing_files:
        print("all required files found: setup complete!")
        return

    missing_files_str = "\t" + "\n\t".join(str(DATA_DIR / x) for x in missing_files)

    response = input(
        f"detected missing setup files:\n"
        f"{missing_files_str}\n"
        f"do you want to download them from {WEB_ADDRESS}?\n"
        f"[y/N]\n"
    )
    download_allowed = response.strip().lower() in ["y", "yes"]

    if not download_allowed:
        print("download denied")
        return 

    print("dowloading files...")

    for file_name in missing_files:
        download_link = WEB_ADDRESS + file_name + DOWNLOAD_SUFFIX
        download_from_link(download_link, DATA_DIR)


def download_from_link(url, target_dir):
    parsed_url = urlparse(url)
    raw_filename = os.path.basename(parsed_url.path)

    if not raw_filename:
        raw_filename = "downloaded_file"

    os.makedirs(target_dir, exist_ok=True)
    local_filename = os.path.join(target_dir, raw_filename)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    formatted_size = decimal(total_size)

    print(f"Downloading {url} ({formatted_size})")
    progress_bar = Progress(
        BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",
        "•", DownloadColumn(),
        "•", TransferSpeedColumn(),
        "•", TimeRemainingColumn(),
    )

    with progress_bar as progress:
        task = progress.add_task("   ", total=total_size)

        with open(local_filename, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                progress.update(task, advance=len(data))


if __name__ == "__main__":
    main()
