import numpy as np
import pandas as pd

from script_loader import load_script

clean_and_sort = load_script("01_clean_and_sort.py")


def test_process_directory_skips_empty_files_and_processes_csvs(
    tmp_path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.csv").write_text(
        "participant,Alter,Geschlecht (m, w, d),date\n1,30,w,2026-01-01\n"
    )
    (input_dir / "empty.csv").write_text("")

    out_dir = tmp_path / "out"
    result = clean_and_sort.process_directory(str(input_dir), str(out_dir))

    assert result == 0
    assert (out_dir / "0001_gamble.csv").exists()
    assert (out_dir / "0001_questionnaire.csv").exists()
    assert not (out_dir / "empty_gamble.csv").exists()
    assert not (out_dir / "empty_questionnaire.csv").exists()


def test_process_file_writes_expected_gamble_columns(tmp_path) -> None:
    csv_path = tmp_path / "1234_test.csv"
    pd.DataFrame(
        {
            "teil_eins.started": ["start1"],
            "teil_eins.stopped": ["stop1"],
            "key_resp_one.keys": ["1"],
            "key_resp_one.rt": ["0.25"],
            "trials.thisN": ["10"],
            "gain1": ["20"],
            "loss1": ["5"],
            "teil_zwei.started": ["start2"],
            "teil_zwei.stopped": ["stop2"],
            "key_resp_two.keys": ["2"],
            "key_resp_two.rt": ["0.30"],
            "trials_2.thisN": ["11"],
            "gain2": ["30"],
            "loss2": ["6"],
            "teil_drei.started": ["start3"],
            "teil_drei.stopped": ["stop3"],
            "key_resp_three.keys": ["3"],
            "key_resp_three.rt": ["0.35"],
            "trials_3.thisN": ["12"],
            "gain3": ["40"],
            "loss3": ["7"],
            "participant": ["1234"],
            "Alter": ["30"],
            "Geschlecht": ["w"],
            "date": ["2026-01-01"],
        }
    ).to_csv(csv_path, index=False)

    clean_and_sort.process_file(str(csv_path), str(tmp_path))

    output_path = tmp_path / "1234_gamble.csv"
    assert output_path.exists()
    output_df = pd.read_csv(output_path)
    assert list(output_df.columns) == [
        "started",
        "stopped",
        "key_resp",
        "reaction_time",
        "block_count",
        "gain",
        "loss",
    ]


def test_process_file_computes_reaction_time_from_started_and_stopped(
    tmp_path,
) -> None:
    csv_path = tmp_path / "9999_test.csv"
    pd.DataFrame(
        {
            "teil_eins.started": ["10.0"],
            "teil_eins.stopped": ["12.5"],
            "key_resp_one.keys": ["1"],
            "trials.thisN": ["10"],
            "gain1": ["20"],
            "loss1": ["5"],
            "teil_zwei.started": ["20.0"],
            "teil_zwei.stopped": ["23.0"],
            "key_resp_two.keys": ["2"],
            "trials_2.thisN": ["11"],
            "gain2": ["30"],
            "loss2": ["6"],
            "teil_drei.started": ["30.0"],
            "teil_drei.stopped": ["34.2"],
            "key_resp_three.keys": ["3"],
            "trials_3.thisN": ["12"],
            "gain3": ["40"],
            "loss3": ["7"],
            "participant": ["9999"],
            "Alter": ["30"],
            "Geschlecht": ["w"],
            "date": ["2026-01-01"],
        }
    ).to_csv(csv_path, index=False)

    clean_and_sort.process_file(str(csv_path), str(tmp_path))

    output_path = tmp_path / "9999_gamble.csv"
    assert output_path.exists()
    output_df = pd.read_csv(output_path)
    non_missing_rt = output_df["reaction_time"].dropna().tolist()
    assert np.allclose(non_missing_rt[:3], [2.5, 3.0, 4.2])


def test_extract_questionnaire_uses_last_non_empty_value() -> None:
    df = pd.DataFrame(
        {
            "key": [
                "participant",
                "survey.block_1/question1.I have healthy boundaries.",
                "survey.block_1/question1.I have healthy boundaries.",
                "survey.block_1/question3.I have healthy boundaries.",
            ],
            "value": ["1234", "", "5", "7"],
        }
    )

    result = clean_and_sort.extract_questionnaire(
        df,
        keycol="key",
        valcol="value",
        expected_long_keys=[
            "survey.block_1/question1.I have healthy boundaries."
        ],
    )

    assert result["participant"] == "1234"
    assert result["b1_q1_boundaries"] == "5"


def test_process_file_reads_wide_questionnaire_columns_with_trailing_empty_rows(
    tmp_path,
) -> None:
    csv_path = tmp_path / "10195_test.csv"
    pd.DataFrame(
        {
            "participant": ["10195", None],
            "Alter": ["23", None],
            "Geschlecht (m, w, d)": ["w", None],
            "date": ["2026-06-23", None],
            "survey.block_1/question1.I have healthy boundaries.": [
                "Agree low",
                None,
            ],
            "survey.block_1/question3.I have healthy boundaries.": [
                "Disagree low",
                None,
            ],
            "survey.block_1/question4.I have healthy boundaries.": [
                "Agree moderate",
                None,
            ],
        }
    ).to_csv(csv_path, index=False)

    clean_and_sort.process_file(str(csv_path), str(tmp_path))

    questionnaire_path = tmp_path / "10195_questionnaire.csv"
    questionnaire_df = pd.read_csv(questionnaire_path)
    assert questionnaire_df.loc[0, "b1_q1_boundaries"] == 3
    assert questionnaire_df.loc[0, "b1_q3_boundaries"] == 2
    assert questionnaire_df.loc[0, "b1_q4_boundaries"] == 4


def test_process_file_writes_gender_column(tmp_path) -> None:
    csv_path = tmp_path / "5678_test.csv"
    pd.DataFrame(
        {
            "participant": ["5678"],
            "Alter": ["31"],
            "Geschlecht (m, w, d)": ["w"],
            "date": ["2026-01-02"],
        }
    ).to_csv(csv_path, index=False)

    clean_and_sort.process_file(str(csv_path), str(tmp_path))

    questionnaire_path = tmp_path / "5678_questionnaire.csv"
    assert questionnaire_path.exists()
    questionnaire_df = pd.read_csv(questionnaire_path)
    assert "gender" in questionnaire_df.columns
    assert questionnaire_df.loc[0, "gender"] == "w"
    assert "age" in questionnaire_df.columns


def test_normalize_gender_value_standardizes_expected_values() -> None:
    assert clean_and_sort.normalize_gender_value("weiblich") == "w"
    assert clean_and_sort.normalize_gender_value("männlich") == "m"
    assert clean_and_sort.normalize_gender_value("d") == "d"


def test_normalize_gender_value_handles_invalid_numeric_values() -> None:
    assert clean_and_sort.normalize_gender_value("29") is None
