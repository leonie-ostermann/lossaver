"""Step 3: Create per-participant diagnostic scatter plots from processed gamble data.

Reads all `<participant-id>_gamble.csv` files from `processed_data/` and, for each
participant, creates two model-free scatter plots:

- Expectancy value plot: `expectancy_value = gain - loss` (x-axis) vs. the
  observed response `key_resp` (y-axis, values 1-4).
- Gain-vs-loss plot: `gain` (x-axis) vs. `loss` (y-axis), with each point
  color-coded by the observed response `key_resp`.

Both plots visualize the same raw response pattern that the logistic
regression in step 2 summarizes numerically, without any model assumptions.

Usage:
    python scripts/03_participant_plots.py --input-dir processed_data --plots-dir plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

RESPONSE_LABELS = {
    1: "Disagree strong",
    2: "Disagree",
    3: "Agree",
    4: "Agree strong",
}

RESPONSE_COLORS = {
    1: "#8B0000",  # very red
    2: "#F4A6A6",  # light red
    3: "#90EE90",  # light green
    4: "#006400",  # very green
}

GAIN_LOSS_RESPONSE_LABELS = {
    1: "1 = Disagree strong",
    2: "2 = Disagree",
    3: "3 = Agree",
    4: "4 = Agree strong",
}


def add_trial_count_badge(ax: plt.Axes, n_trials: int) -> None:
    ax.text(
        0.99,
        1.01,
        f"n={n_trials}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        zorder=5,
        clip_on=False,
    )


def format_participant_id_grid(
    participant_ids: List[str], columns: int = 8
) -> str:
    """Lay participant IDs out in aligned columns so they print like a table."""
    width = max(len(pid) for pid in participant_ids)
    rows = [
        participant_ids[i : i + columns]
        for i in range(0, len(participant_ids), columns)
    ]
    return "\n".join(
        "  " + "  ".join(pid.ljust(width) for pid in row) for row in rows
    )


def participant_id_from_frame_or_path(df: pd.DataFrame, csv_path: Path) -> str:
    if "participant" in df.columns:
        participant_values = (
            df["participant"].astype(str).replace("", pd.NA).dropna()
        )
        if not participant_values.empty:
            return str(participant_values.iloc[-1]).strip()
    return csv_path.stem.replace("_gamble", "")


# --- Expectancy value plot ------------------------------------------------


def build_expectancy_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"gain", "loss", "key_resp"}
    if not required_columns.issubset(df.columns):
        missing = sorted(required_columns.difference(df.columns))
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = pd.DataFrame()
    frame["gain"] = pd.to_numeric(df["gain"], errors="coerce")
    frame["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    frame["key_resp"] = pd.to_numeric(df["key_resp"], errors="coerce")
    frame["expectancy_value"] = frame["gain"] - frame["loss"]
    frame = frame.dropna(subset=["expectancy_value", "key_resp"])
    frame = frame[frame["key_resp"].isin(RESPONSE_LABELS.keys())]
    return frame


def plot_single_participant_expectancy(
    file_path: str, output_path: str, skipped_files: List[str] | None = None
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(file_path)
    df = pd.read_csv(csv_path, dtype=str)
    frame = build_expectancy_frame(df)
    if frame.empty:
        participant_id = csv_path.stem.replace("_gamble", "")
        if skipped_files is not None:
            skipped_files.append(participant_id)
        else:
            print(
                f"Skipping {file_path}: no usable rows with key_resp in 1..4"
            )
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        frame["expectancy_value"],
        frame["key_resp"],
        color="C0",
        edgecolor="black",
        s=38,
        alpha=0.85,
    )

    participant_id = participant_id_from_frame_or_path(df, csv_path)
    ax.set_title(f"Expectancy value vs response for {participant_id}")
    ax.set_xlabel("Expectancy value (gain - loss)")
    ax.set_ylabel("key_resp")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0.75, 4.25)
    ax.grid(True, alpha=0.3)
    add_trial_count_badge(ax, n_trials=len(frame))

    legend_handles: List[mlines.Line2D] = [
        mlines.Line2D([], [], color="none", label=f"{key} = {label}")
        for key, label in RESPONSE_LABELS.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        title="Response coding",
    )

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_expectancy_for_directory(
    input_dir: str,
    plots_dir: str = "plots",
    skipped_files: List[str] | None = None,
) -> Path:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    # When the caller does not supply an accumulator list, collect and print locally
    # so the function keeps working the same way when used on its own.
    print_locally = skipped_files is None
    if skipped_files is None:
        skipped_files = []

    csv_files = sorted(input_path.glob("*_gamble.csv"))
    total = len(csv_files)
    width = len(str(total))
    plotted_count = 0
    for i, csv_path in enumerate(csv_files, start=1):
        participant_id = csv_path.stem.replace("_gamble", "")
        print(
            f"[{i:>{width}}/{total}] Plotting expectancy value: {participant_id}"
        )
        output_path = plots_path / f"{participant_id}_expectancy_value.png"
        plot_single_participant_expectancy(
            str(csv_path), str(output_path), skipped_files=skipped_files
        )
        if output_path.exists():
            plotted_count += 1

    if plotted_count == 0:
        raise FileNotFoundError(
            f"No participant files with plottable rows found in {input_dir}"
        )

    if print_locally and skipped_files:
        print("Info: no usable rows with key_resp in 1..4, skipped for:")
        print(format_participant_id_grid(skipped_files))

    return plots_path


# --- Gain vs loss plot -----------------------------------------------------


def build_gain_loss_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"gain", "loss", "key_resp"}
    if not required_columns.issubset(df.columns):
        missing = sorted(required_columns.difference(df.columns))
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = pd.DataFrame()
    frame["gain"] = pd.to_numeric(df["gain"], errors="coerce")
    frame["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    frame["key_resp"] = pd.to_numeric(df["key_resp"], errors="coerce")
    frame = frame.dropna(subset=["gain", "loss", "key_resp"])
    frame = frame[frame["key_resp"].isin(RESPONSE_COLORS.keys())]
    frame["key_resp"] = frame["key_resp"].astype(int)
    return frame


def plot_single_participant_gain_loss(
    file_path: str, output_path: str, skipped_files: List[str] | None = None
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(file_path)
    df = pd.read_csv(csv_path, dtype=str)
    frame = build_gain_loss_frame(df)
    if frame.empty:
        participant_id = csv_path.stem.replace("_gamble", "")
        if skipped_files is not None:
            skipped_files.append(participant_id)
        else:
            print(
                f"Skipping {file_path}: no usable rows with key_resp in 1..4"
            )
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for response_value in [1, 2, 3, 4]:
        subset = frame[frame["key_resp"] == response_value]
        if subset.empty:
            continue
        ax.scatter(
            subset["gain"],
            subset["loss"],
            color=RESPONSE_COLORS[response_value],
            edgecolor="black",
            s=42,
            alpha=0.9,
            label=GAIN_LOSS_RESPONSE_LABELS[response_value],
        )

    participant_id = participant_id_from_frame_or_path(df, csv_path)
    ax.set_title(f"Gain vs Loss responses for {participant_id}")
    ax.set_xlabel("gain")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    add_trial_count_badge(ax, n_trials=len(frame))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        title="key_resp",
    )

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_gain_loss_for_directory(
    input_dir: str,
    plots_dir: str = "plots",
    skipped_files: List[str] | None = None,
) -> Path:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    # When the caller does not supply an accumulator list, collect and print locally
    # so the function keeps working the same way when used on its own.
    print_locally = skipped_files is None
    if skipped_files is None:
        skipped_files = []

    csv_files = sorted(input_path.glob("*_gamble.csv"))
    total = len(csv_files)
    width = len(str(total))
    plotted_count = 0
    for i, csv_path in enumerate(csv_files, start=1):
        participant_id = csv_path.stem.replace("_gamble", "")
        print(
            f"[{i:>{width}}/{total}] Plotting gain-loss response: {participant_id}"
        )
        output_path = plots_path / f"{participant_id}_gain_loss_response.png"
        plot_single_participant_gain_loss(
            str(csv_path), str(output_path), skipped_files=skipped_files
        )
        if output_path.exists():
            plotted_count += 1

    if plotted_count == 0:
        raise FileNotFoundError(
            f"No participant files with plottable rows found in {input_dir}"
        )

    if print_locally and skipped_files:
        print("Info: no usable rows with key_resp in 1..4, skipped for:")
        print(format_participant_id_grid(skipped_files))

    return plots_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create participant-level expectancy-value and gain-vs-loss scatter plots"
    )
    parser.add_argument(
        "--input-dir",
        default="processed_data",
        help="Directory containing processed *_gamble.csv files (default: processed_data)",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory where plots are written (default: plots)",
    )
    args = parser.parse_args()

    expectancy_skipped: List[str] = []
    gain_loss_skipped: List[str] = []
    expectancy_plots_path = plot_expectancy_for_directory(
        args.input_dir, args.plots_dir, skipped_files=expectancy_skipped
    )
    gain_loss_plots_path = plot_gain_loss_for_directory(
        args.input_dir, args.plots_dir, skipped_files=gain_loss_skipped
    )

    print(f"Saved expectancy-value plots to {expectancy_plots_path}")
    print(f"Saved gain-loss response plots to {gain_loss_plots_path}")

    if expectancy_skipped:
        print(
            "Info: no usable rows with key_resp in 1..4, skipped for expectancy-value plot:"
        )
        print(format_participant_id_grid(expectancy_skipped))

    if gain_loss_skipped:
        print(
            "Info: no usable rows with key_resp in 1..4, skipped for gain-loss plot:"
        )
        print(format_participant_id_grid(gain_loss_skipped))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
