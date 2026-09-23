import pandas as pd

from script_loader import load_script

merge_questionnaires = load_script("04_merge_questionnaires.py")


def test_merge_writes_questionnaire_values_as_integer_text(tmp_path) -> None:
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "participant": ["10195", "10196"],
            "beta_gain": [1.2, 1.1],
            "beta_loss": [-0.8, -0.7],
            "lambda": [0.67, 0.64],
        }
    ).to_csv(summary_path, index=False)

    questionnaire_dir = tmp_path / "questionnaires"
    questionnaire_dir.mkdir()
    pd.DataFrame(
        {
            "participant": ["10195"],
            "age": [23],
            "gender": ["w"],
            "date": ["2026-06-23"],
            "b1_q1_boundaries": [3],
        }
    ).to_csv(questionnaire_dir / "10195_questionnaire.csv", index=False)

    output_path = tmp_path / "extended.csv"
    merge_questionnaires.merge_summary_with_questionnaires(
        str(summary_path), str(questionnaire_dir), str(output_path)
    )

    output_text = output_path.read_text()
    assert "23.0" not in output_text
    assert "3.0" not in output_text

    merged_df = pd.read_csv(output_path, dtype=str)
    assert merged_df.loc[0, "age"] == "23"
    assert merged_df.loc[0, "b1_q1_boundaries"] == "3"
