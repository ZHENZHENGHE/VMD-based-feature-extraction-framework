# -*- coding: utf-8 -*-
"""
04_plot_results.py

我用这个脚本画 event-guided 与 fixed-window 的分类结果对比图。
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(r"D:/a_work/课题组实验数据处理/新预处理/results/merged_ml/classification_results")
summary = pd.read_csv(DATA_DIR / "event_vs_fixed_ml_summary.csv")

# 我先选择每个 Method-Model 的最佳 AUC 结果，避免同一个模型不同 k_features 重复显示。
best = (
    summary
    .sort_values(["Method", "Model", "AUC", "Balanced_ACC"], ascending=[True, True, False, False])
    .groupby(["Method", "Model"], as_index=False)
    .head(1)
)

metrics = ["ACC", "SEN", "SPE", "F1", "AUC"]

for metric in metrics:
    pivot = best.pivot(index="Model", columns="Method", values=metric)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(f"Event-guided vs fixed-window classification: {metric}")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(DATA_DIR / f"classification_{metric}.png", dpi=300)
    plt.show()

print(best[["Method", "Model", "N_Features_Selected", "ACC", "SEN", "SPE", "F1", "AUC"]])
