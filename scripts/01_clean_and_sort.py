"""Step 1: Clean and align raw Mixed Gamble CSV exports.

Reads the raw PsychoPy/PsychoJS export files (one CSV per participant)
from `raw_data/` and produces two tidy CSV files per participant in
`processed_data/`:

- `<participant-id>_gamble.csv`:
    128 trial rows with columns started, stopped, key_resp,
    reaction_time, block_count, gain, loss
- `<participant-id>_questionnaire.csv`:
    one row with participant, age, gender, date and the 30
    short-coded survey items (b<block>_q<question>_<phrase>)

Usage:
    python scripts/01_clean_and_sort.py <input_csv> [--out-dir processed_data]
    python scripts/01_clean_and_sort.py --input-dir raw_data \
        --out-dir processed_data
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

BLOCK_DEFS = [
    ("teil_eins.started", "loss1", 43),
    ("teil_zwei.started", "loss2", 43),
    ("teil_drei.started", "loss3", 42),
]

# Phrase -> short token used for questionnaire item column names
PHRASE_TOKENS = {
    "I have healthy boundaries": "boundaries",
    "I have a lot of self-care": "selfcare",
    (
        "I have a healthy dose of self-respect, and don’t let "
        "people take advantage of me"
    ): "selfrespect",
    "I balance my own needs with the needs of others": "balance",
    "I advocate for my own needs": "advocate",
    (
        "I have a healthy form of selfishness (e.g., meditation, "
        "eating healthy, exercising, etc.) that doesn’t hurt "
        "others, but brings me greater happiness"
    ): "healthy_selfish",
    "Even though I give a lot to others, I know when to recharge": "recharge",
    (
        "I give myself permission to enjoy myself, even if it "
        "doesn’t necessarily help others"
    ): "permission",
    "I take good care of myself": "takecare",
    (
        "I prioritize my own personal projects over the demands " "of others"
    ): "prioritize",
}

RESPONSE_CODES = {
    "Disagree strong": 0,
    "Disagree moderate": 1,
    "Disagree low": 2,
    "Agree low": 3,
    "Agree moderate": 4,
    "Agree strong": 5,
}


def find_key_value_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Heuristic to find the key/value columns in a long-format export."""
    cols = [c for c in df.columns]
    if "key" in cols and "value" in cols:
        return "key", "value"
    if len(cols) >= 2:
        return cols[0], cols[1]
    return cols[0], cols[0]


def extract_block_values(
    df: pd.DataFrame, keycol: str, valcol: str, start_key: str, end_key: str
) -> List[str]:
    s = df[keycol].astype(str)
    start_mask = s.str.contains(start_key, na=False)
    end_mask = s.str.contains(end_key, na=False)
    if not start_mask.any() or not end_mask.any():
        return []
    start_idx = start_mask.idxmax()
    end_idx = list(df[end_mask].index)[-1]
    block = df.loc[start_idx:end_idx]
    mask_not_markers = ~(
        block[keycol].astype(str).str.contains(start_key, na=False)
        | block[keycol].astype(str).str.contains(end_key, na=False)
    )
    values = block.loc[mask_not_markers, valcol].astype(str).tolist()
    return values


def align_blocks(blocks: List[List[str]]) -> pd.DataFrame:
    max_len = max(len(b) for b in blocks)
    data = {}
    for i, b in enumerate(blocks, start=1):
        col = f"block{i}"
        series = b + [None] * (max_len - len(b))
        data[col] = series
    df = pd.DataFrame(data)
    df.index = df.index + 1
    df.index.name = "trial"
    return df


def short_key_from_long(long_key: str) -> Optional[str]:
    m = re.match(r"survey\.block_(\d+)/question(\d+)\.(.+)$", long_key)
    if not m:
        return None
    block, qnum, phrase = m.group(1), m.group(2), m.group(3)
    phrase = phrase.strip()
    phrase_norm = phrase.rstrip(". ,\"')")
    token = (
        PHRASE_TOKENS.get(phrase)
        or PHRASE_TOKENS.get(phrase_norm)
        or PHRASE_TOKENS.get(phrase.replace(",", ""))
    )
    if not token:
        token = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")[:24]
    return f"b{block}_q{qnum}_{token}"


def last_non_empty_value(series: pd.Series) -> Optional[str]:
    cleaned = (
        series.astype(str).replace("", pd.NA).replace("nan", pd.NA).dropna()
    )
    if cleaned.empty:
        return None
    return cleaned.iloc[-1]


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_matching_survey_column(
    df: pd.DataFrame, question_label: str, phrase: str
) -> Optional[str]:
    prefix = f"survey.block_1/{question_label}."
    phrase_norm = normalize_column_name(phrase)
    for col in df.columns:
        col_name = str(col)
        if not col_name.startswith(prefix):
            continue
        if normalize_column_name(col_name).endswith(phrase_norm):
            return col_name
    return None


def extract_questionnaire(
    df: pd.DataFrame, keycol: str, valcol: str, expected_long_keys: List[str]
) -> Dict[str, Optional[str]]:
    s = df[keycol].astype(str)
    out: Dict[str, Optional[str]] = {}
    for var in ["participant", "Alter", "Geschlecht", "date"]:
        mask = s.str.contains(var, na=False, case=False, regex=False)
        if mask.any():
            out[var] = last_non_empty_value(df.loc[mask, valcol])
        else:
            out[var] = None

    for long_key in expected_long_keys:
        mask = s.eq(long_key)
        if not mask.any():
            mask = s.str.contains(long_key, na=False, regex=False)
        if not mask.any():
            val = None
        else:
            val = last_non_empty_value(df.loc[mask, valcol])
        short = short_key_from_long(long_key)
        if short:
            out[short] = val
    return out


def map_response_value(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    return RESPONSE_CODES.get(value)


def normalize_gender_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value_norm = str(value).strip().lower()
    if value_norm in {"", "nan", "none", "na", "n/a"}:
        return None

    mapping = {
        "w": "w",
        "weiblich": "w",
        "f": "w",
        "female": "w",
        "m": "m",
        "männlich": "m",
        "maennlich": "m",
        "male": "m",
        "d": "d",
        "divers": "d",
        "diverse": "d",
    }
    if value_norm in mapping:
        return mapping[value_norm]

    # Some exports use a numeric coding instead of text.
    if value_norm == "1":
        return "m"
    if value_norm == "2":
        return "w"
    if value_norm == "3":
        return "d"

    # Unknown/invalid value (e.g. an accidental age number) is
    # treated as missing.
    return None


def process_file(path: str, out_dir: str = "processed_data") -> None:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path_obj.stat().st_size == 0:
        print(f"Skipping empty file: {path}")
        return

    df = pd.read_csv(path, header=0, dtype=str)
    keycol, valcol = find_key_value_columns(df)

    # Target columns for the stacked trial table (128 rows total).
    TARGET_COLS = [
        "started",
        "stopped",
        "key_resp",
        "reaction_time",
        "block_count",
        "gain",
        "loss",
    ]

    block_defs = [
        {
            "start_col": "teil_eins.started",
            "stop_col": "teil_eins.stopped",
            "end_col": "loss1",
            "key_resp_prefix": "key_resp_one",
            "trials_prefix": "trials",
            "gain_col": "gain1",
            "loss_col": "loss1",
            "expected_rows": 43,
        },
        {
            "start_col": "teil_zwei.started",
            "stop_col": "teil_zwei.stopped",
            "end_col": "loss2",
            "key_resp_prefix": "key_resp_two",
            "trials_prefix": "trials_2",
            "gain_col": "gain2",
            "loss_col": "loss2",
            "expected_rows": 43,
        },
        {
            "start_col": "teil_drei.started",
            "stop_col": "teil_drei.stopped",
            "end_col": "loss3",
            "key_resp_prefix": "key_resp_three",
            "trials_prefix": "trials_3",
            "gain_col": "gain3",
            "loss_col": "loss3",
            "expected_rows": 42,
        },
    ]

    block_tables: List[pd.DataFrame] = []
    for b in block_defs:
        if b["start_col"] not in df.columns or b["end_col"] not in df.columns:
            block_tables.append(
                pd.DataFrame(
                    [{k: None for k in TARGET_COLS}] * b["expected_rows"]
                )
            )
            continue

        start_idx = df.columns.get_loc(b["start_col"])
        end_idx = df.columns.get_loc(b["end_col"])
        block_cols = list(df.columns[start_idx : end_idx + 1])
        block_df = df.loc[:, block_cols].copy()

        non_empty_mask = block_df.apply(
            lambda col: col.notna() & col.astype(str).str.strip().ne("")
        ).any(axis=1)
        block_rows = block_df.loc[non_empty_mask].copy()

        if len(block_rows) > b["expected_rows"]:
            block_rows = block_rows.iloc[: b["expected_rows"]]
        elif len(block_rows) < b["expected_rows"]:
            missing_rows = b["expected_rows"] - len(block_rows)
            if missing_rows > 0:
                block_rows = pd.concat(
                    [
                        block_rows,
                        pd.DataFrame(
                            [{col: None for col in block_cols}] * missing_rows
                        ),
                    ],
                    ignore_index=True,
                )

        rows: List[Dict[str, Optional[str]]] = []
        for _, row in block_rows.iterrows():
            started_value = row[b["start_col"]]
            stopped_value = row[b["stop_col"]]
            started_numeric = pd.to_numeric(
                pd.Series([started_value]), errors="coerce"
            ).iloc[0]
            stopped_numeric = pd.to_numeric(
                pd.Series([stopped_value]), errors="coerce"
            ).iloc[0]
            reaction_time: Optional[float]
            if pd.isna(started_numeric) or pd.isna(stopped_numeric):
                reaction_time = None
            else:
                reaction_time = float(stopped_numeric - started_numeric)

            out: Dict[str, Optional[str]] = {
                "started": started_value,
                "stopped": stopped_value,
                "key_resp": row[f"{b['key_resp_prefix']}.keys"],
                "reaction_time": reaction_time,
                "block_count": row[f"{b['trials_prefix']}.thisN"],
                "gain": row[b["gain_col"]],
                "loss": row[b["loss_col"]],
            }
            rows.append(out)

        block_tables.append(pd.DataFrame(rows, columns=TARGET_COLS))

    stacked = pd.concat(block_tables, ignore_index=True)
    if len(stacked) < 128:
        missing = 128 - len(stacked)
        stacked = pd.concat(
            [
                stacked,
                pd.DataFrame([{k: None for k in TARGET_COLS}] * missing),
            ],
            ignore_index=True,
        )
    elif len(stacked) > 128:
        stacked = stacked.iloc[:128].reset_index(drop=True)
    else:
        stacked = stacked.reset_index(drop=True)

    basename = os.path.basename(path)
    m = re.match(r"(\d{1,6})_", basename)
    if m:
        pid = m.group(1).zfill(4)
    else:
        pid = os.path.splitext(basename)[0]

    participant_value = None
    if "participant" in df.columns:
        series = df["participant"].astype(str).replace("", pd.NA).dropna()
        if not series.empty:
            participant_value = series.iloc[-1]
    if participant_value is not None and participant_value != "nan":
        participant_value = str(participant_value).strip()
        if re.fullmatch(r"\d+", participant_value):
            pid = participant_value.zfill(4)
        else:
            pid = participant_value

    os.makedirs(out_dir, exist_ok=True)
    gamble_path = os.path.join(out_dir, f"{pid}_gamble.csv")
    stacked.to_csv(gamble_path, index=False)

    # Build the list of expected long questionnaire column keys.
    expected_long_keys: List[str] = []
    phrases = list(PHRASE_TOKENS.keys())
    for qn in ["question1", "question3", "question4"]:
        for ph in phrases:
            matched_col = find_matching_survey_column(df, qn, ph)
            if matched_col:
                expected_long_keys.append(matched_col)
            else:
                expected_long_keys.append(f"survey.block_1/{qn}.{ph}")

    q: Dict[str, Optional[str]]
    survey_cols_present = any(
        str(col).startswith("survey.block_1/") for col in df.columns
    )
    q = {}

    for var in ["participant", "Alter", "Geschlecht", "date"]:
        if var in df.columns:
            q[var] = last_non_empty_value(df[var])
        else:
            q[var] = None

    gender_col = next(
        (
            col
            for col in df.columns
            if str(col).lower() == "geschlecht (m, w, d)"
        ),
        None,
    )
    if gender_col is not None:
        q["Geschlecht"] = last_non_empty_value(df[gender_col])
    elif (
        "Geschlecht" in q
        and q.get("Geschlecht") is None
        and "Geschlecht (m, w, d)" in df.columns
    ):
        q["Geschlecht"] = last_non_empty_value(df["Geschlecht (m, w, d)"])

    if survey_cols_present:
        for long_key in expected_long_keys:
            if long_key in df.columns:
                q_val = last_non_empty_value(df[long_key])
            else:
                q_val = None
            short = short_key_from_long(long_key)
            if short:
                q[short] = map_response_value(q_val)
    else:
        fallback_q = extract_questionnaire(
            df, keycol, valcol, expected_long_keys
        )
        for key, value in fallback_q.items():
            if key in {"participant", "Alter", "Geschlecht", "date"}:
                if q.get(key) is None:
                    q[key] = value
            elif key.startswith("b") and value is not None:
                q[key] = map_response_value(value)
            else:
                q[key] = value

    out_row = {
        "participant": q.get("participant"),
        "age": q.get("Alter"),
        "gender": normalize_gender_value(q.get("Geschlecht")),
        "date": q.get("date"),
    }
    for long_key in expected_long_keys:
        short = short_key_from_long(long_key)
        if short:
            out_row[short] = q.get(short)

    questionnaire_path = os.path.join(out_dir, f"{pid}_questionnaire.csv")
    pd.DataFrame([out_row]).to_csv(questionnaire_path, index=False)


def process_directory(input_dir: str, out_dir: str = "processed_data") -> int:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return 0

    total = len(csv_files)
    width = len(str(total))
    for i, csv_path in enumerate(csv_files, start=1):
        progress = f"[{i:>{width}}/{total}]"
        if csv_path.stat().st_size == 0:
            print(f"{progress} Skipping empty file: {csv_path}")
            continue
        print(f"{progress} Processing {csv_path.name} -> {out_dir}")
        process_file(str(csv_path), out_dir)
    print(f"Finished processing {total} file(s) from {input_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean and align raw Mixed Gamble CSV exports"
    )
    parser.add_argument(
        "input_csv", nargs="?", help="Single raw CSV file to process"
    )
    parser.add_argument(
        "--input-dir", help="Directory containing raw CSV files to process"
    )
    parser.add_argument(
        "--out-dir",
        default="processed_data",
        help="Output directory (default: processed_data)",
    )
    args = parser.parse_args()

    if args.input_dir:
        return process_directory(args.input_dir, args.out_dir)
    if args.input_csv:
        process_file(args.input_csv, args.out_dir)
        return 0

    parser.error("Provide either an input CSV file or --input-dir")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
