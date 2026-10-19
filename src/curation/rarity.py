import json
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Global variable to hold fingerprints across worker processes efficiently
_SHARED_FPS = []


def _init_worker(fps):
    global _SHARED_FPS
    _SHARED_FPS = fps


def _compute_chunk(chunk_data):
    """Computes similarity neighbor counts for a chunk of molecule indices."""
    global _SHARED_FPS
    chunk_id, chunk_indices = chunk_data
    results = {}
    for i in chunk_indices:
        target_fp = _SHARED_FPS[i]
        similarities = DataStructs.BulkDiceSimilarity(target_fp, _SHARED_FPS)
        # Count matches >= 0.66, excluding self-comparison
        match_count = sum(
            1 for idx, sim in enumerate(similarities) if idx != i and sim >= 0.66
        )
        results[i] = match_count
    return results


def compute_np_atlas_rarity(
    input_jsonl,
    output_jsonl,
    dice_threshold=0.66,
    radius=2,
    n_bits=2048,
):
    print("Step 1: Loading data and generating Morgan fingerprints...")
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    entries = []
    fps = []

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            smiles = data.get("smiles")

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = morgan_gen.GetFingerprint(mol)
                entries.append(data)
                fps.append(fp)

            if (idx + 1) % 50000 == 0:
                print(f"Processed {idx + 1} lines...")

    total_mols = len(entries)
    print(f"Successfully processed {total_mols} valid molecules.")

    num_cores = max(1, cpu_count() - 1)
    print(
        f"Step 2: Calculating similarity neighbor counts in parallel using {num_cores} CPU cores..."
    )

    # Split indices into fine-grained chunks for smooth progress updates
    indices = list(range(total_mols))
    num_chunks = min(len(indices), num_cores * 40)
    chunks = np.array_split(indices, num_chunks)
    chunk_tuples = [(i, list(chunk)) for i, chunk in enumerate(chunks)]

    rarity_scores = np.zeros(total_mols, dtype=int)

    # Run multiprocessing pool with a Rich Progress context manager
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(
            "Processing similarities...", total=len(chunk_tuples)
        )

        with Pool(
            processes=num_cores, initializer=_init_worker, initargs=(fps,)
        ) as pool:
            for chunk_res in pool.imap_unordered(_compute_chunk, chunk_tuples):
                for idx, count in chunk_res.items():
                    rarity_scores[idx] = count
                progress.update(task_id, advance=1)

    elapsed_total = time.time() - start_time
    print(f"\nStep 2 completed in {elapsed_total / 60:.2f} minutes.")

    print("Step 3: Saving results with rarity scores...")
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, data in enumerate(entries):
            data["neighbor_count"] = int(rarity_scores[idx])
            data["rarity_score"] = float(1.0 / (1.0 + rarity_scores[idx]))
            f_out.write(json.dumps(data) + "\n")

    print(f"Done! Enriched file saved to {output_jsonl}")


if __name__ == "__main__":
    DB = Path("data/coconut_db.jsonl")
    NEW_FILE = DB.parent / f"{DB.stem}_enriched{DB.suffix}"
    compute_np_atlas_rarity(DB, NEW_FILE)
