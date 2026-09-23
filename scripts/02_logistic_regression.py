"""Step 2: Fit a per-participant logistic regression of accept/reject.

Reads all `<participant-id>_gamble.csv` files from `processed_data/`
and, for each participant, fits `key_resp in {3,4} ~ gain + loss`
(accept vs. reject) using an unregularized logistic regression. Writes:

- `plots/<participant-id>_logistic_regression.png`:
    fitted curve plot
- `plots/<participant-id>_logistic_regression_coefficients.csv`:
    intercept/beta_gain/beta_loss
- `logistic_regression_summary.csv`:
    one row per participant with beta_bias, beta_gain, beta_loss, lambda

Usage:
    python scripts/02_logistic_regression.py <single_gamble_csv> \
        [--output plot.png]
    python scripts/02_logistic_regression.py --input-dir processed_data \
        --summary-output logistic_regression_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def logistic(values: np.ndarray | list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values_array))


def format_participant_id_grid(
    participant_ids: List[str], columns: int = 8
) -> str:
    """Lay participant IDs out in aligned columns, like a table."""
    width = max(len(pid) for pid in participant_ids)
    rows = [
        participant_ids[i : i + columns]
        for i in range(0, len(participant_ids), columns)
    ]
    return "\n".join(
        "  " + "  ".join(pid.ljust(width) for pid in row) for row in rows
    )


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


def map_response_to_binary(values: pd.Series) -> pd.Series:
    mapping = {1: 0, 2: 0, 3: 1, 4: 1}
    mapped = values.map(mapping)
    return mapped.where(values.isin(mapping.keys()), pd.NA)


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[["gain", "loss"]].copy()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame["target"] = map_response_to_binary(
        pd.to_numeric(df["key_resp"], errors="coerce")
    )
    return frame.dropna(subset=["target", "gain", "loss"])


def fit_logistic_regression(df: pd.DataFrame) -> LogisticRegression:
    training_frame = build_training_frame(df)
    if training_frame.empty:
        raise ValueError("No usable rows available for logistic regression")

    X = training_frame[["gain", "loss"]]
    y = training_frame["target"]
    if y.nunique() < 2:
        raise ValueError(
            "Logistic regression requires both classes in the target"
        )

    # penalty=None gives a plain maximum-likelihood fit; lbfgs supports
    # unregularized estimation natively.
    model = LogisticRegression(penalty=None, max_iter=1000, solver="lbfgs")
    model.fit(X, y)
    return model


def collect_clean_data(input_dir: str) -> pd.DataFrame:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    frames: List[pd.DataFrame] = []
    for csv_path in sorted(input_path.glob("*_gamble.csv")):
        frame = pd.read_csv(csv_path)
        if "key_resp" in frame.columns:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No processed gamble CSV files found in {input_dir}"
        )

    return pd.concat(frames, ignore_index=True)


def summarize_participants(
    input_dir: str,
    non_positive_gain_participants: List[str] | None = None,
    non_negative_loss_participants: List[str] | None = None,
    both_non_positive_gain_and_non_negative_loss: List[str] | None = None,
) -> pd.DataFrame:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # When the caller does not supply accumulator lists, collect and print locally
    # so the function keeps working the same way when used on its own.
    print_locally = non_positive_gain_participants is None
    if non_positive_gain_participants is None:
        non_positive_gain_participants = []
    if non_negative_loss_participants is None:
        non_negative_loss_participants = []
    if both_non_positive_gain_and_non_negative_loss is None:
        both_non_positive_gain_and_non_negative_loss = []

    rows: List[dict[str, float | str | None]] = []
    csv_files = sorted(input_path.glob("*_gamble.csv"))
    total = len(csv_files)
    width = len(str(total))
    for i, csv_path in enumerate(csv_files, start=1):
        print(
            f"[{i:>{width}}/{total}] Fitting logistic regression: "
            f"{csv_path.name}"
        )
        frame = pd.read_csv(csv_path, dtype=str)
        if (
            "key_resp" not in frame.columns
            or "gain" not in frame.columns
            or "loss" not in frame.columns
        ):
            continue

        participant_value = None
        if "participant" in frame.columns:
            series = (
                frame["participant"].astype(str).replace("", pd.NA).dropna()
            )
            if not series.empty:
                participant_value = series.iloc[-1]
        if participant_value is None:
            participant_value = csv_path.stem.replace("_gamble", "")
        if participant_value is not None:
            participant_value = str(participant_value)

        try:
            model = fit_logistic_regression(frame)
        except ValueError:
            continue

        intercept = float(model.intercept_[0])
        coefficients = model.coef_[0]
        beta_gain = float(coefficients[0])
        beta_loss = float(coefficients[1])

        if beta_gain <= 0 and beta_loss >= 0:
            both_non_positive_gain_and_non_negative_loss.append(
                participant_value
            )
        elif beta_gain <= 0:
            non_positive_gain_participants.append(participant_value)
        elif beta_loss >= 0:
            non_negative_loss_participants.append(participant_value)

        # Using the absolute value of beta_loss makes lambda robust to
        # positive or negative sign conventions for losses in the raw data.
        lambda_value = abs(beta_loss) / beta_gain

        rows.append(
            {
                "participant": str(participant_value),
                "beta_bias": intercept,
                "beta_gain": beta_gain,
                "beta_loss": beta_loss,
                "lambda": lambda_value,
            }
        )

    if not rows:
        raise FileNotFoundError(
            "No participant gamble files with valid parameters "
            f"were found in {input_dir}"
        )

    if print_locally:
        if both_non_positive_gain_and_non_negative_loss:
            print(
                "Info: non-positive gain slope and non-negative loss "
                "slope included in summary for:"
            )
            print(
                format_participant_id_grid(
                    both_non_positive_gain_and_non_negative_loss
                )
            )

        if non_positive_gain_participants:
            print("Info: non-positive gain slope included in summary for:")
            print(format_participant_id_grid(non_positive_gain_participants))

        if non_negative_loss_participants:
            print("Info: non-negative loss slope included in summary for:")
            print(format_participant_id_grid(non_negative_loss_participants))

    return pd.DataFrame(rows)


def plot_lambda(summary: pd.DataFrame, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(summary))
    ax.bar(x, summary["lambda"].tolist(), color="steelblue")
    ax.axhline(
        1.0,
        color="firebrick",
        linestyle="--",
        linewidth=1,
        label="Loss Neutrality (\u03bb=1)",
    )
    ax.set_title("Estimated lambda per participant (Cleaned Data)")
    ax.set_xlabel("Participant")
    ax.set_ylabel("lambda = |beta_loss| / beta_gain")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        summary["participant"].tolist(), rotation=45, ha="right"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_single_participant(
    file_path: str,
    output_path: str,
    non_positive_gain_files: List[str] | None = None,
    no_target_files: List[str] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(file_path, dtype=str)
    if (
        "key_resp" not in frame.columns
        or "gain" not in frame.columns
        or "loss" not in frame.columns
    ):
        raise ValueError(f"{file_path} does not contain the expected columns")

    try:
        model = fit_logistic_regression(frame)
    except ValueError:
        participant_id = Path(file_path).stem.replace("_gamble", "")
        if no_target_files is not None:
            no_target_files.append(participant_id)
        else:
            print(
                f"Skipping {file_path}: no usable target values "
                "for logistic regression"
            )
        return

    intercept = float(model.intercept_[0])
    beta_gain = float(model.coef_[0, 0])
    beta_loss = float(model.coef_[0, 1])

    if beta_gain <= 0:
        participant_id = Path(file_path).stem.replace("_gamble", "")
        if non_positive_gain_files is not None:
            non_positive_gain_files.append(participant_id)
        else:
            print(
                "Info: non-positive gain slope in this individual "
                f"model for: {participant_id}"
            )

    coefficients_frame = pd.DataFrame(
        {
            "parameter": ["intercept", "beta_gain", "beta_loss"],
            "value": [intercept, beta_gain, beta_loss],
        }
    )
    coefficients_path = output.with_name(f"{output.stem}_coefficients.csv")
    coefficients_frame.to_csv(coefficients_path, index=False)

    training_frame = build_training_frame(frame)
    X = training_frame[["gain", "loss"]].to_numpy()

    fixed_loss = float(np.median(X[:, 1]))
    gain_min, gain_max = np.min(X[:, 0]), np.max(X[:, 0])
    gain_grid = np.linspace(gain_min, gain_max, 200)
    linear_predictor = (
        intercept + beta_gain * gain_grid + beta_loss * fixed_loss
    )
    probabilities = logistic(linear_predictor)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(gain_grid, probabilities, color="C0", linewidth=2.0)
    ax.scatter(
        training_frame["gain"],
        training_frame["target"],
        color="C1",
        edgecolor="black",
        s=40,
        zorder=3,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Logistic function for {Path(file_path).stem}")
    ax.set_xlabel("gain")
    ax.set_ylabel("P(accept)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    add_trial_count_badge(ax, n_trials=len(training_frame))
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_participants_in_directory(
    input_dir: str,
    plots_dir: str = "plots",
    non_positive_gain_files: List[str] | None = None,
    no_target_files: List[str] | None = None,
) -> Path:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    # When the caller does not supply accumulator lists, collect and print
    # locally so the function keeps working the same way when used on its own.
    print_locally = non_positive_gain_files is None
    if non_positive_gain_files is None:
        non_positive_gain_files = []
    if no_target_files is None:
        no_target_files = []

    csv_files = sorted(input_path.glob("*_gamble.csv"))
    total = len(csv_files)
    width = len(str(total))
    for i, csv_path in enumerate(csv_files, start=1):
        stem = csv_path.stem.replace("_gamble", "")
        print(f"[{i:>{width}}/{total}] Plotting logistic regression: {stem}")
        output_path = plots_path / f"{stem}_logistic_regression.png"
        plot_single_participant(
            str(csv_path),
            str(output_path),
            non_positive_gain_files=non_positive_gain_files,
            no_target_files=no_target_files,
        )

    if print_locally:
        if no_target_files:
            print(
                "Info: no usable target values for logistic "
                "regression, skipped for:"
            )
            print(format_participant_id_grid(no_target_files))

        if non_positive_gain_files:
            print(
                "Info: non-positive gain slope in the individual "
                "model for:"
            )
            print(format_participant_id_grid(non_positive_gain_files))

    return plots_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit participant-level logistic regressions on cleaned "
            "gamble data"
        )
    )
    parser.add_argument(
        "input_file", nargs="?", help="Single cleaned gamble CSV file to plot"
    )
    parser.add_argument(
        "--input-dir", help="Directory containing cleaned *_gamble.csv files"
    )
    parser.add_argument(
        "--output",
        help="Optional explicit output path for the single-file plot",
    )
    parser.add_argument(
        "--summary-output",
        default="logistic_regression_summary.csv",
        help="Path for the regression summary CSV",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory for per-participant plots (default: plots)",
    )
    args = parser.parse_args()

    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        stem = input_path.stem.replace("_gamble", "")
        output_path = (
            Path(args.output)
            if args.output
            else input_path.with_name(f"{stem}_logistic_regression.png")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_single_participant(str(input_path), str(output_path))
        print(f"Saved single-participant plot to {output_path}")
        return 0

    if not args.input_dir:
        parser.error("Provide either an input CSV file or --input-dir")
        return 2

    summary_output_path = Path(args.summary_output)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    non_positive_gain_participants: List[str] = []
    non_negative_loss_participants: List[str] = []
    both_non_positive_gain_and_non_negative_loss: List[str] = []
    summary = summarize_participants(
        args.input_dir,
        non_positive_gain_participants=non_positive_gain_participants,
        non_negative_loss_participants=non_negative_loss_participants,
        both_non_positive_gain_and_non_negative_loss=both_non_positive_gain_and_non_negative_loss,
    )
    summary.to_csv(summary_output_path, index=False)

    non_positive_gain_files: List[str] = []
    no_target_files: List[str] = []
    plots_path = plot_participants_in_directory(
        args.input_dir,
        plots_dir=args.plots_dir,
        non_positive_gain_files=non_positive_gain_files,
        no_target_files=no_target_files,
    )

    print(f"Saved regression summary to {summary_output_path}")
    print(f"Saved participant plots to {plots_path}")

    if both_non_positive_gain_and_non_negative_loss:
        print(
            "Info: non-positive gain slope and non-negative loss slope included in summary for:"
        )
        print(
            format_participant_id_grid(
                both_non_positive_gain_and_non_negative_loss
            )
        )

    if non_positive_gain_participants:
        print("Info: non-positive gain slope included in summary for:")
        print(format_participant_id_grid(non_positive_gain_participants))

    if non_negative_loss_participants:
        print("Info: non-negative loss slope included in summary for:")
        print(format_participant_id_grid(non_negative_loss_participants))

    if no_target_files:
        print(
            "Info: no usable target values for logistic regression, skipped for:"
        )
        print(format_participant_id_grid(no_target_files))

    if non_positive_gain_files:
        print("Info: non-positive gain slope in the individual model for:")
        print(format_participant_id_grid(non_positive_gain_files))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
