import pandas as pd

from script_loader import load_script

participant_plots = load_script("03_participant_plots.py")


def test_build_expectancy_frame_filters_to_valid_key_resp_values() -> None:
    df = pd.DataFrame(
        {
            "gain": [10, 20, 30, 40, 50],
            "loss": [5, 10, 15, 20, 25],
            "key_resp": [1, 2, 3, 4, 5],
        }
    )

    frame = participant_plots.build_expectancy_frame(df)
    assert frame["key_resp"].tolist() == [1, 2, 3, 4]
    assert frame["expectancy_value"].tolist() == [5, 10, 15, 20]


def test_build_gain_loss_frame_filters_to_valid_key_resp_values() -> None:
    df = pd.DataFrame(
        {
            "gain": [10, 20, 30, 40, 50],
            "loss": [5, 10, 15, 20, 25],
            "key_resp": [1, 2, 3, 4, 5],
        }
    )

    frame = participant_plots.build_gain_loss_frame(df)
    assert frame["key_resp"].tolist() == [1, 2, 3, 4]
    assert frame["gain"].tolist() == [10, 20, 30, 40]
    assert frame["loss"].tolist() == [5, 10, 15, 20]


def test_plot_expectancy_for_directory_writes_participant_plot(
    tmp_path,
) -> None:
    input_dir = tmp_path / "processed"
    input_dir.mkdir()

    pd.DataFrame(
        {
            "participant": ["001", "001", "001", "001"],
            "gain": [10, 20, 30, 40],
            "loss": [5, 10, 35, 20],
            "key_resp": [1, 2, 3, 4],
        }
    ).to_csv(input_dir / "001_gamble.csv", index=False)

    plots_dir = tmp_path / "plots"
    result = participant_plots.plot_expectancy_for_directory(
        str(input_dir), str(plots_dir)
    )

    assert result == plots_dir
    assert (plots_dir / "001_expectancy_value.png").exists()


def test_plot_gain_loss_for_directory_writes_participant_plot(
    tmp_path,
) -> None:
    input_dir = tmp_path / "processed"
    input_dir.mkdir()

    pd.DataFrame(
        {
            "participant": ["001", "001", "001", "001"],
            "gain": [10, 20, 30, 40],
            "loss": [5, 10, 35, 20],
            "key_resp": [1, 2, 3, 4],
        }
    ).to_csv(input_dir / "001_gamble.csv", index=False)

    plots_dir = tmp_path / "plots"
    result = participant_plots.plot_gain_loss_for_directory(
        str(input_dir), str(plots_dir)
    )

    assert result == plots_dir
    assert (plots_dir / "001_gain_loss_response.png").exists()
