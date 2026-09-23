#!/usr/bin/env bash
#
# Sets up the Python virtual environment and runs the full lossaver pipeline
# (scripts 01-04) from raw data to the final participant summary.
#
# Usage:
#   ./run_pipeline.sh [raw_data_dir] [processed_data_dir] [plots_dir]
#
# All arguments are optional and default to raw_data/, processed_data/, plots/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RAW_DATA_DIR="${1:-raw_data}"
PROCESSED_DATA_DIR="${2:-processed_data}"
PLOTS_DIR="${3:-plots}"
SUMMARY_CSV="logistic_regression_summary.csv"
PARTICIPANT_SUMMARY_CSV="participant_summary.csv"
VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Setting up virtual environment in $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

echo "==> Installing dependencies from requirements-dev.txt"
"$VENV_PYTHON" -m pip install -U pip
"$VENV_PYTHON" -m pip install -r requirements-dev.txt

echo "==> Step 1/4: Cleaning and aligning raw data ($RAW_DATA_DIR -> $PROCESSED_DATA_DIR)"
"$VENV_PYTHON" scripts/01_clean_and_sort.py --input-dir "$RAW_DATA_DIR" --out-dir "$PROCESSED_DATA_DIR"

echo "==> Step 2/4: Fitting per-participant logistic regressions"
"$VENV_PYTHON" scripts/02_logistic_regression.py \
    --input-dir "$PROCESSED_DATA_DIR" \
    --summary-output "$SUMMARY_CSV" \
    --plots-dir "$PLOTS_DIR"

echo "==> Step 3/4: Creating participant diagnostic plots"
"$VENV_PYTHON" scripts/03_participant_plots.py --input-dir "$PROCESSED_DATA_DIR" --plots-dir "$PLOTS_DIR"

echo "==> Step 4/4: Merging regression results with questionnaire data"
"$VENV_PYTHON" scripts/04_merge_questionnaires.py \
    --summary "$SUMMARY_CSV" \
    --quest-dir "$PROCESSED_DATA_DIR" \
    --output "$PARTICIPANT_SUMMARY_CSV"

echo "==> Done. Final participant summary: $PARTICIPANT_SUMMARY_CSV"
