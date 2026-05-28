# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

from ppvmd_ml_utils import (
    build_subject_level_tables,
    check_subject_table,
)

ROOT_DIR = Path(r"D:/a_work/课题组实验数据处理/新预处理/results")
OUT_DIR = ROOT_DIR / "merged_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 在这里手动固定每个受试者的真实标签
# 0 = healthy
# 1 = STC / patient
# ============================================================

SUBJECT_LABELS = {
    "2bHT000005": 0,
    "2bHT000007": 0,
    "2cHT000001": 0,
    "2cHT000009": 0,

    "3bSY000146": 1,
    "3cSY000128": 1,
    "22SY000117": 1,
    "27HT000003": 0,

    "27HT000008": 0,
    "27JD000001": 0,
    "32SY000109": 1,

    "32SY000131": 1,
    "37SY000145": 1,

    "51SY000110": 1,
    "56SY000111": 1,
}


def read_feature_table(path: Path) -> pd.DataFrame:
    """根据文件后缀自动读取 xlsx / xls / csv。"""

    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {path}")


event_tables = []
fixed_tables = []
label_records = []

for subject_dir in ROOT_DIR.iterdir():

    if not subject_dir.is_dir():
        continue

    if subject_dir.name in ["merged_ml", "paper_outputs", "merged"]:
        continue

    subject_id = subject_dir.name

    if subject_id not in SUBJECT_LABELS:
        print(f"Skip {subject_id}: label not defined.")
        continue

    label = SUBJECT_LABELS[subject_id]

    main_dir = subject_dir / "main"

    if not main_dir.exists():
        print(f"Skip {subject_id}: no main folder.")
        continue

    event_candidates = list(
        main_dir.glob(f"{subject_id}_event_guided_phase_features.*")
    )

    fixed_candidates = list(
        main_dir.glob(f"{subject_id}_fixed_phase_features.*")
    )

    if len(event_candidates) == 0:
        print(f"Missing event table: {subject_id}")
    else:
        event_path = event_candidates[0]
        print(f"Read event table: {event_path.name}")

        df_event = read_feature_table(event_path)

        # 强制覆盖原表里的 SubjectID 和 Label，避免旧文件残留错误标签。
        df_event["SubjectID"] = subject_id
        df_event["Label"] = label
        df_event["WindowMethod"] = "event_guided"

        event_tables.append(df_event)

    if len(fixed_candidates) == 0:
        print(f"Missing fixed table: {subject_id}")
    else:
        fixed_path = fixed_candidates[0]
        print(f"Read fixed table: {fixed_path.name}")

        df_fixed = read_feature_table(fixed_path)

        df_fixed["SubjectID"] = subject_id
        df_fixed["Label"] = label
        df_fixed["WindowMethod"] = "fixed"

        fixed_tables.append(df_fixed)

    label_records.append(
        {
            "SubjectID": subject_id,
            "Label": label,
            "HasEventTable": len(event_candidates) > 0,
            "HasFixedTable": len(fixed_candidates) > 0,
        }
    )


if len(event_tables) == 0:
    raise RuntimeError("No event-guided feature tables found.")

if len(fixed_tables) == 0:
    raise RuntimeError("No fixed-window feature tables found.")


event_window_df = pd.concat(event_tables, ignore_index=True)
fixed_window_df = pd.concat(fixed_tables, ignore_index=True)

print("\nWindow-level tables")
print("Event window-level shape:", event_window_df.shape)
print("Fixed window-level shape:", fixed_window_df.shape)

print("\nEvent label distribution:")
print(event_window_df[["SubjectID", "Label"]].drop_duplicates().sort_values("SubjectID"))

print("\nFixed label distribution:")
print(fixed_window_df[["SubjectID", "Label"]].drop_duplicates().sort_values("SubjectID"))


event_window_df.to_csv(
    OUT_DIR / "all_event_window_level_features.csv",
    index=False,
    encoding="utf-8-sig",
)

fixed_window_df.to_csv(
    OUT_DIR / "all_fixed_window_level_features.csv",
    index=False,
    encoding="utf-8-sig",
)

label_audit_df = pd.DataFrame(label_records).sort_values("SubjectID")
label_audit_df.to_csv(
    OUT_DIR / "subject_label_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

event_subject_df, fixed_subject_df = build_subject_level_tables(
    event_window_df=event_window_df,
    fixed_window_df=fixed_window_df,
    output_dir=OUT_DIR,
    aggregations=("mean", "std", "median", "iqr"),
)

print("\nSubject-level tables")
print("Event subject-level shape:", event_subject_df.shape)
print("Fixed subject-level shape:", fixed_subject_df.shape)

event_qc = check_subject_table(event_subject_df)
fixed_qc = check_subject_table(fixed_subject_df)

event_qc.to_csv(
    OUT_DIR / "event_subject_table_qc.csv",
    index=False,
    encoding="utf-8-sig",
)

fixed_qc.to_csv(
    OUT_DIR / "fixed_subject_table_qc.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nEvent QC:")
print(event_qc)

print("\nFixed QC:")
print(fixed_qc)

print("\nSaved to:", OUT_DIR.resolve())