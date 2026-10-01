import argparse
import random
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

REPO = Path(__file__).parent.parent.parent.resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly remove a specified percentage of entries from a JSONL structure database."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input JSONL database file.",
    )
    parser.add_argument(
        "-p",
        "--percent",
        type=float,
        required=True,
        help="Percentage of entries to remove (0.0 to 100.0).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to the output JSONL file. Defaults to <input_name>_<percent>pct_masked.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility (default: 42).",
    )

    args = parser.parse_args()

    if not 0.0 <= args.percent <= 100.0:
        parser.error("--percent must be between 0.0 and 100.0")

    return args


def generate_default_output_path(input_path: Path, percent: float) -> Path:
    # Formats percentage without trailing zeros if integer (e.g. 20 instead of 20.0)
    formatted_pct = f"{percent:.2f}".rstrip("0").rstrip(".")
    new_stem = f"{input_path.stem}_{formatted_pct}pct_masked"
    return input_path.with_name(f"{new_stem}{input_path.suffix}")


def mask_structure_db(
    input_file: str | Path,
    percent_to_remove: float,
    output_file: str | Path | None = None,
    seed: int = 42,
) -> None:
    input_path = REPO / Path(input_file)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_file is None:
        output_path = REPO / generate_default_output_path(input_path, percent_to_remove)
    else:
        output_path = REPO / Path(output_file)

    # Initialize Random Number Generator instance
    rng = random.Random(seed)

    print(f"Reading records from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        records = [line for line in f if line.strip()]

    total_count = len(records)
    keep_ratio = 1.0 - (percent_to_remove / 100.0)
    target_count = int(round(total_count * keep_ratio))

    print(
        f"Total records: {total_count:,} | Removing: {percent_to_remove}% | Keeping: {target_count:,} | Seed: {seed}"
    )

    # Randomly select lines to keep using dedicated RNG instance
    retained_records = rng.sample(records, target_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
    )

    with progress_bar as progress, open(output_path, "w", encoding="utf-8") as f_out:
        task = progress.add_task("Writing output database...", total=target_count)
        for record in retained_records:
            f_out.write(record)
            progress.update(task, advance=1)

    print(f"Masked JSONL database saved to: {output_path}")


def main():
    args = parse_args()
    mask_structure_db(
        input_file=args.input,
        percent_to_remove=args.percent,
        output_file=args.output,
        seed=args.seed,
    )
    print()


if __name__ == "__main__":
    main()
