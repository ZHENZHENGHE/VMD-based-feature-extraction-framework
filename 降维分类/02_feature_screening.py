# -*- coding: utf-8 -*-
"""
02_feature_screening.py

用这个脚本分别对 event-guided 和 fixed-window 受试者级表做单变量特征筛选。
这些结果主要用于解释和候选特征排序，不直接替代交叉验证内部的特征筛选。
"""

from pathlib import Path
import pandas as pd

from ppvmd_ml_utils import univariate_feature_screening


DATA_DIR = Path(r"D:/a_work/课题组实验数据处理/新预处理/results/merged_ml")

for name, file_name in [
    ("event_guided", "all_event_subject_features.csv"),
    ("fixed", "all_fixed_subject_features.csv"),
]:
    df = pd.read_csv(DATA_DIR / file_name)
    stat_df = univariate_feature_screening(df)
    stat_df.to_csv(DATA_DIR / f"{name}_univariate_feature_screening.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(name)
    print(stat_df.head(30))
