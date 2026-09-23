"""Step 4: Merge the logistic-regression summary with per-participant questionnaire data.

Reads `logistic_regression_summary.csv` (from step 2) and all
`<participant-id>_questionnaire.csv` files (from step 1), joins them on
`participant`, and writes the combined table to `participant_summary.csv`.

Usage:
    python scripts/04_merge_questionnaires.py --summary logistic_regression_summary.csv \
        --quest-dir processed_data --output participant_summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd

QUESTIONNAIRE_INTEGER_COLUMNS = re.compile(r"^b\d+_")


def collect_questionnaire_data(
    input_dir: str, read_warnings: List[str] | None = None
) -> pd.DataFrame:
    """Find all *_questionnaire.csv files and combine them into one DataFrame."""
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(
            f"Questionnaire directory not found: {input_dir}"
        )

    # When the caller does not supply an accumulator list, collect and print locally
    # so the function keeps working the same way when used on its own.
    print_locally = read_warnings is None
    if read_warnings is None:
        read_warnings = []

    frames: List[pd.DataFrame] = []
    csv_files = sorted(input_path.glob("*_questionnaire.csv"))
    total = len(csv_files)
    width = len(str(total))
    for i, csv_path in enumerate(csv_files, start=1):
        print(f"[{i:>{width}}/{total}] Reading questionnaire: {csv_path.name}")
        try:
            df = pd.read_csv(csv_path)

            # If the 'participant' column is missing or empty, derive the ID from the
            # file name (e.g. '15715_questionnaire.csv' -> '15715').
            if (
                "participant" not in df.columns
                or df["participant"].dropna().empty
            ):
                participant_id = csv_path.stem.replace("_questionnaire", "")
                df["participant"] = participant_id

            df["participant"] = df["participant"].astype(str).str.strip()
            frames.append(df)

        except Exception as e:
            read_warnings.append(f"{csv_path.name}: {e}")

    if not frames:
        raise FileNotFoundError(
            f"No valid *_questionnaire.csv files found in {input_dir}."
        )

    if print_locally and read_warnings:
        print("Warning: could not read the following files:")
        for warning in read_warnings:
            print(f"  - {warning}")

    combined_questionnaires = pd.concat(frames, ignore_index=True)
    return combined_questionnaires


def normalize_questionnaire_integer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert questionnaire scale items and age to nullable integer columns."""
    normalized = df.copy()
    for col in normalized.columns:
        if col == "age" or QUESTIONNAIRE_INTEGER_COLUMNS.match(col):
            numeric = pd.to_numeric(normalized[col], errors="coerce")
            normalized[col] = numeric.astype("Int64")
    return normalized


def merge_summary_with_questionnaires(
    summary_csv_path: str, questionnaire_dir: str, output_csv_path: str
) -> None:
    """Join the regression summary with the questionnaire data."""
    summary_path = Path(summary_csv_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_csv_path}")

    print(f"Loading regression summary: {summary_path.name}")
    summary_df = pd.read_csv(summary_path)
    summary_df["participant"] = (
        summary_df["participant"].astype(str).str.strip()
    )

    print(f"Collecting questionnaire data from directory: {questionnaire_dir}")
    read_warnings: List[str] = []
    questionnaire_df = collect_questionnaire_data(
        questionnaire_dir, read_warnings=read_warnings
    )

    print("Merging data sets...")
    # Left join keeps every participant from the summary and adds questionnaire columns.
    merged_df = pd.merge(
        summary_df, questionnaire_df, on="participant", how="left"
    )
    merged_df = normalize_questionnaire_integer_columns(merged_df)

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Saved successfully to: {output_path}")

    if read_warnings:
        print("Warning: could not read the following files:")
        for warning in read_warnings:
            print(f"  - {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extend the logistic-regression summary with questionnaire data."
    )
    parser.add_argument(
        "--summary",
        default="logistic_regression_summary.csv",
        help="Path to the existing logistic_regression_summary.csv",
    )
    parser.add_argument(
        "--quest-dir",
        required=True,
        help="Directory containing the *_questionnaire.csv files",
    )
    parser.add_argument(
        "--output",
        default="participant_summary.csv",
        help="Path for the final combined CSV file",
    )
    args = parser.parse_args()

    try:
        merge_summary_with_questionnaires(
            summary_csv_path=args.summary,
            questionnaire_dir=args.quest_dir,
            output_csv_path=args.output,
        )
        return 0
    except Exception as e:
        print(f"Error during execution: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
