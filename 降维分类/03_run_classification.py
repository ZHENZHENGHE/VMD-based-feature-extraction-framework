# -*- coding: utf-8 -*-
"""
03_run_classification.py

用这个脚本运行最终分类实验：
1. event-guided 受试者级特征分类
2. fixed-window 受试者级特征分类
3. 多分类器横向比较
4. 输出 ACC/SEN/SPE/PPV/NPV/F1/AUC

注意：
在交叉验证内部完成缺失值填充、标准化和特征筛选，避免数据泄漏。
"""

from pathlib import Path
import pandas as pd

from ppvmd_ml_utils import run_event_fixed_ml_experiment


DATA_DIR = Path(r"D:/a_work/课题组实验数据处理/新预处理/results/merged_ml")
OUT_DIR = DATA_DIR / "classification_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


event_subject_df = pd.read_csv(DATA_DIR / "all_event_subject_features.csv")
fixed_subject_df = pd.read_csv(DATA_DIR / "all_fixed_subject_features.csv")


summary_df, pred_df = run_event_fixed_ml_experiment(
    event_subject_df=event_subject_df,
    fixed_subject_df=fixed_subject_df,
    output_dir=OUT_DIR,
    random_state=42,
    preferred_splits=5,
    # k_features_list=(10, 20, 30),
    k_features_list=[10],
    scaler="standard",
    feature_selection="kbest_f",
)

print("Classification summary:")
print(summary_df)
print("\nSaved to:", OUT_DIR)
