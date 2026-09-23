import numpy as np
import pandas as pd

from script_loader import load_script

logistic_regression = load_script("02_logistic_regression.py")


def test_map_response_to_binary_converts_expected_values() -> None:
    values = pd.Series([1, 2, 3, 4, 5])
    result = logistic_regression.map_response_to_binary(values)
    assert result.iloc[:4].tolist() == [0, 0, 1, 1]
    assert pd.isna(result.iloc[4])


def test_logistic_returns_sigmoid_values() -> None:
    values = logistic_regression.logistic(np.array([0.0, 5.0]))
    assert np.isclose(values[0], 0.5)
    assert values[1] > 0.99


def test_build_training_frame_uses_numeric_features() -> None:
    df = pd.DataFrame(
        {
            "key_resp": [1, 2, 3, 4],
            "gain": [1.0, 2.0, 3.0, 4.0],
            "loss": [4.0, 3.0, 2.0, 1.0],
        }
    )

    frame = logistic_regression.build_training_frame(df)
    assert list(frame.columns) == ["gain", "loss", "target"]
    assert frame["target"].tolist() == [0, 0, 1, 1]


def test_summarize_participants_returns_coefficients(tmp_path) -> None:
    input_dir = tmp_path / "processed"
    input_dir.mkdir()
    pd.DataFrame(
        {
            "key_resp": [1, 2, 3, 4],
            "gain": [1.0, 2.0, 3.0, 4.0],
            "loss": [4.0, 3.0, 2.0, 1.0],
            "participant": ["001", "001", "001", "001"],
        }
    ).to_csv(input_dir / "001_gamble.csv", index=False)
    pd.DataFrame(
        {
            "key_resp": [None, None],
            "gain": [1.0, 2.0],
            "loss": [3.0, 4.0],
            "participant": ["002", "002"],
        }
    ).to_csv(input_dir / "002_gamble.csv", index=False)

    summary = logistic_regression.summarize_participants(str(input_dir))
    assert list(summary.columns) == [
        "participant",
        "beta_bias",
        "beta_gain",
        "beta_loss",
        "lambda",
    ]
    assert summary.loc[0, "participant"] == "001"
    assert "lambda" in summary.columns


def test_plot_participants_in_directory_writes_individual_plots(
    tmp_path,
) -> None:
    input_dir = tmp_path / "processed"
    input_dir.mkdir()
    pd.DataFrame(
        {
            "key_resp": [1, 2, 3, 4],
            "gain": [1.0, 2.0, 3.0, 4.0],
            "loss": [4.0, 3.0, 2.0, 1.0],
            "participant": ["001", "001", "001", "001"],
        }
    ).to_csv(input_dir / "001_gamble.csv", index=False)

    plots_dir = tmp_path / "plots"
    result = logistic_regression.plot_participants_in_directory(
        str(input_dir), str(plots_dir)
    )

    assert result == plots_dir
    assert (plots_dir / "001_logistic_regression.png").exists()
    assert (plots_dir / "001_logistic_regression_coefficients.csv").exists()
