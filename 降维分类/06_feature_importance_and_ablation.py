# -*- coding: utf-8 -*-
"""
06_feature_importance_and_ablation.py

用这个脚本完成两个适合小样本医学数据的解释性分析：

1. Feature group ablation
   特征组消融实验：
   分别只使用 Entropy / RQA / Fractal / PhaseSpace / LLE /
   VMD_Mode3 / VMD_Mode4 / CoreNonlinear / AllBiomarker 特征做 LOSO 分类，
   用来判断哪一类动力学信息最有判别价值。

2. LOSO selection frequency
   LOSO 特征稳定性筛选：
   在每一个 Leave-One-Subject-Out 训练折内部执行 SelectKBest，
   然后统计每个特征在所有折中被选中的频率。
   这个适合当前 15 例左右的小样本医学数据。

重要修改：
- 剔除了 Tau / EmbeddingDim / EmbeddedVectors / RQA_epsilon / NumWindows 等算法参数或窗口结构变量，
  避免把它们错误解释为生理 biomarker。
- 所有分类性能评估均使用 subject-level LOSO。
- 每一折内部完成 imputation / scaling / feature selection，避免信息泄漏。
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(r"D:/a_work/课题组实验数据处理/新预处理/results")
IN_DIR = ROOT_DIR / "merged_ml"
OUT_DIR = ROOT_DIR / "ml_validation_loso"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_FILE = IN_DIR / "all_event_subject_features.csv"
FIXED_FILE = IN_DIR / "all_fixed_subject_features.csv"


# ============================================================
# Global parameters
# ============================================================

RF_N_ESTIMATORS = 200
RF_MIN_SAMPLES_LEAF = 2

K_ABLATION = 10
K_IMPORTANCE = 20

STABLE_FREQ_THRESHOLD = 0.60

RANDOM_STATE = 42


# ============================================================
# Feature filtering rules
# ============================================================

# 把这些变量视为算法参数、窗口结构变量或潜在非生理变量，不作为主 biomarker。
NON_BIOMARKER_PATTERNS = [
    "NumWindows",
    "WindowID",
    "WindowMethod",
    "StartTime",
    "EndTime",
    "CenterTime",
    "StartIndex",
    "EndIndex",
    "EventID",
    "EventStartTime",
    "EventEndTime",
    "EventCoverage",
    "NumPoints",
    "Tau",
    "EmbeddingDim",
    "EmbeddedVectors",
    "RQA_epsilon",
    "epsilon",
]


def is_non_biomarker_feature(feature_name: str) -> bool:
    """判断一个特征是否应从主 biomarker 分析中剔除。"""

    return any(pattern in feature_name for pattern in NON_BIOMARKER_PATTERNS)


def infer_feature_group(feature_name: str) -> str:
    """根据特征名给每个特征标注所属动力学类别。"""

    if "RQA" in feature_name:
        return "RQA"

    if "SampEn" in feature_name or "PermEn" in feature_name:
        return "Entropy"

    if "HFD" in feature_name or "KFD" in feature_name or "PFD" in feature_name:
        return "Fractal"

    if "PS_" in feature_name:
        return "PhaseSpace"

    if "ApproxLLE" in feature_name:
        return "LLE"

    if (
        "Skewness" in feature_name
        or "Kurtosis" in feature_name
        or "Diff_MAD" in feature_name
        or "Diff_Energy" in feature_name
    ):
        return "DistributionMorphology"

    return "Other"


# ============================================================
# Basic utilities
# ============================================================

def load_table(path: Path):
    """读取 subject-level 表，并只保留可用于建模的数值型 biomarker 特征。"""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    if "SubjectID" not in df.columns:
        raise ValueError(f"SubjectID column not found in {path}")

    if "Label" not in df.columns:
        raise ValueError(f"Label column not found in {path}")

    y = df["Label"].astype(int).values
    sid = df["SubjectID"].astype(str).values

    X_df = df.drop(columns=["SubjectID", "Label"], errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number])

    # 删除全空列和常数列。
    X_df = X_df.dropna(axis=1, how="all")
    X_df = X_df.loc[:, X_df.nunique(dropna=True) > 1]

    # 删除算法参数、窗口结构变量和不适合作为生理 biomarker 的变量。
    drop_cols = [c for c in X_df.columns if is_non_biomarker_feature(c)]
    X_biomarker = X_df.drop(columns=drop_cols, errors="ignore")

    # 再次删除剔除后可能出现的常数列。
    X_biomarker = X_biomarker.dropna(axis=1, how="all")
    X_biomarker = X_biomarker.loc[:, X_biomarker.nunique(dropna=True) > 1]

    return df, X_biomarker, y, sid, drop_cols


def select_feature_group(X_df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """按照特征名字选择不同非线性特征组。"""

    cols = X_df.columns.tolist()

    if group_name == "Entropy":
        keep = [c for c in cols if infer_feature_group(c) == "Entropy"]

    elif group_name == "RQA":
        keep = [c for c in cols if infer_feature_group(c) == "RQA"]

    elif group_name == "Fractal":
        keep = [c for c in cols if infer_feature_group(c) == "Fractal"]

    elif group_name == "PhaseSpace":
        keep = [c for c in cols if infer_feature_group(c) == "PhaseSpace"]

    elif group_name == "LLE":
        keep = [c for c in cols if infer_feature_group(c) == "LLE"]

    elif group_name == "DistributionMorphology":
        keep = [c for c in cols if infer_feature_group(c) == "DistributionMorphology"]

    elif group_name == "VMD_Mode3":
        keep = [c for c in cols if "VMD_Mode3" in c]

    elif group_name == "VMD_Mode4":
        keep = [c for c in cols if "VMD_Mode4" in c]

    elif group_name == "CoreNonlinear":
        keep = [
            c for c in cols
            if (
                "RQA_DET" in c
                or "RQA_LAM" in c
                or "RQA_L_mean" in c
                or "RQA_L_max" in c
                or "SampEn" in c
                or "PermEn" in c
                or "HFD" in c
                or "KFD" in c
                or "PFD" in c
                or "PS_StateSpread" in c
                or "PS_LogVolume" in c
                or "PS_MeanStep" in c
                or "PS_Eig" in c
            )
        ]

    elif group_name == "AllBiomarker":
        keep = cols

    else:
        raise ValueError(f"Unknown feature group: {group_name}")

    keep = [c for c in keep if c in X_df.columns]
    return X_df[keep].copy()


def calc_metrics(y_true, y_pred, y_score):
    """计算医学分类常用指标。"""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    spe = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    auc = (
        roc_auc_score(y_true, y_score)
        if len(np.unique(y_true)) == 2
        else np.nan
    )

    return {
        "ACC": accuracy_score(y_true, y_pred),
        "SEN": recall_score(y_true, y_pred, zero_division=0),
        "SPE": spe,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": auc,
    }


def make_rf():
    """统一创建随机森林模型，避免不同实验参数不一致。"""

    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# LOSO ablation
# ============================================================

def loso_single_model(X_df, y, model, k_features=10):
    """
    用 LOSO 评估单个特征组的分类性能。
    每一折内部完成 imputation / scaling / SelectKBest / classifier。
    """

    if X_df.shape[1] < 2:
        raise ValueError("At least two features are required.")

    X = X_df.values
    k = min(k_features, X.shape[1])

    loo = LeaveOneOut()

    y_true_all = []
    y_pred_all = []
    y_score_all = []

    for train_idx, test_idx in loo.split(X):

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            ("clf", clone(model)),
        ])

        pipe.fit(X[train_idx], y[train_idx])

        pred = pipe.predict(X[test_idx])

        if hasattr(pipe, "predict_proba"):
            score = pipe.predict_proba(X[test_idx])[:, 1]
        else:
            score = pipe.decision_function(X[test_idx])

        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(pred.tolist())
        y_score_all.extend(score.tolist())

    return calc_metrics(
        np.array(y_true_all),
        np.array(y_pred_all),
        np.array(y_score_all),
    )


def run_feature_group_ablation(datasets):
    """运行 event-guided 和 fixed 的特征组消融实验。"""

    groups = [
        "Entropy",
        "RQA",
        "Fractal",
        "PhaseSpace",
        "LLE",
        "DistributionMorphology",
        "VMD_Mode3",
        "VMD_Mode4",
        "CoreNonlinear",
        "AllBiomarker",
    ]

    model = make_rf()
    rows = []

    for method, (_, X_df, y, sid, drop_cols) in datasets.items():

        print(f"\nRunning ablation for {method}")

        for group in groups:

            Xg = select_feature_group(X_df, group)

            if Xg.shape[1] < 2:
                print(f"  Skip {group}: too few features.")
                continue

            print(f"  Group={group}, n_features={Xg.shape[1]}")

            metrics = loso_single_model(
                Xg,
                y,
                model=model,
                k_features=K_ABLATION,
            )

            metrics.update({
                "Method": method,
                "FeatureGroup": group,
                "NFeatures": Xg.shape[1],
                "SelectedK": min(K_ABLATION, Xg.shape[1]),
                "Model": "RandomForest",
            })

            rows.append(metrics)

    ablation_df = pd.DataFrame(rows)

    ablation_df = ablation_df.sort_values(
        ["Method", "AUC", "ACC"],
        ascending=[True, False, False],
    )

    ablation_df.to_csv(
        OUT_DIR / "feature_group_ablation_loso.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_ablation_results(ablation_df)

    return ablation_df


def plot_ablation_results(ablation_df: pd.DataFrame):
    """绘制特征组消融的 AUC 和 ACC 对比图。"""

    if ablation_df.empty:
        return

    for metric in ["AUC", "ACC"]:

        pivot = ablation_df.pivot(
            index="FeatureGroup",
            columns="Method",
            values=metric,
        )

        # 按照 event_guided 的表现排序，便于观察主方法。
        if "event_guided" in pivot.columns:
            pivot = pivot.sort_values("event_guided", ascending=True)
        else:
            pivot = pivot.sort_values(pivot.columns[0], ascending=True)

        ax = pivot.plot(
            kind="barh",
            figsize=(8, max(4, 0.45 * len(pivot))),
        )

        ax.set_xlabel(metric)
        ax.set_title(f"Feature group ablation based on LOSO {metric}")
        ax.set_xlim(0, 1.0)
        plt.tight_layout()
        plt.savefig(
            OUT_DIR / f"feature_group_ablation_loso_{metric}.png",
            dpi=300,
        )
        plt.close()


# ============================================================
# LOSO selection frequency
# ============================================================

def loso_selection_frequency(X_df, y, method_name, k_features=20):
    """
    统计每个特征在 LOSO 每一折中被 SelectKBest 选中的频率。

    这一步回答：
    哪些特征在不同训练子集里反复被选中。
    """

    X = X_df.values
    k = min(k_features, X.shape[1])

    loo = LeaveOneOut()

    feature_names = X_df.columns.to_numpy()
    selection_counts = pd.Series(0, index=feature_names, dtype=float)

    for fold_id, (train_idx, test_idx) in enumerate(loo.split(X), start=1):

        print(f"  Selection frequency {method_name}: fold {fold_id}/{len(y)}")

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=k)),
        ])

        pipe.fit(X[train_idx], y[train_idx])

        selector = pipe.named_steps["selector"]
        selected = feature_names[selector.get_support()]

        selection_counts.loc[selected] += 1

    freq_df = (
        selection_counts
        .reset_index()
        .rename(columns={"index": "Feature", 0: "SelectedCount"})
    )

    freq_df["SelectionFrequency"] = freq_df["SelectedCount"] / len(y)
    freq_df["Method"] = method_name
    freq_df["SelectedK"] = k
    freq_df["FeatureGroup"] = freq_df["Feature"].apply(infer_feature_group)
    freq_df["IsStable"] = freq_df["SelectionFrequency"] >= STABLE_FREQ_THRESHOLD

    freq_df = freq_df.sort_values(
        ["SelectionFrequency", "SelectedCount"],
        ascending=False,
    )

    out_csv = OUT_DIR / f"{method_name}_loso_selection_frequency.csv"
    freq_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    plot_selection_frequency(freq_df, method_name)

    return freq_df


def plot_selection_frequency(freq_df: pd.DataFrame, method_name: str):
    """绘制 LOSO 特征选择频率前 20 名。"""

    top = freq_df.head(20).iloc[::-1]

    plt.figure(figsize=(8, max(4, 0.35 * len(top))))
    plt.barh(top["Feature"], top["SelectionFrequency"])
    plt.xlabel("Selection frequency across LOSO folds")
    plt.title(f"{method_name}: LOSO feature selection frequency")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"{method_name}_loso_selection_frequency_top20.png",
        dpi=300,
    )
    plt.close()


def summarize_stable_features(all_freq_df: pd.DataFrame):
    """汇总稳定入选特征，并按方法和特征类别统计。"""

    stable_df = all_freq_df[
        all_freq_df["SelectionFrequency"] >= STABLE_FREQ_THRESHOLD
    ].copy()

    stable_df = stable_df.sort_values(
        ["Method", "SelectionFrequency", "FeatureGroup"],
        ascending=[True, False, True],
    )

    stable_df.to_csv(
        OUT_DIR / "stable_loso_selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if len(stable_df) > 0:
        summary = (
            stable_df
            .groupby(["Method", "FeatureGroup"])
            .size()
            .reset_index(name="NStableFeatures")
            .sort_values(["Method", "NStableFeatures"], ascending=[True, False])
        )
    else:
        summary = pd.DataFrame(
            columns=["Method", "FeatureGroup", "NStableFeatures"]
        )

    summary.to_csv(
        OUT_DIR / "stable_feature_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return stable_df, summary


# ============================================================
# QC exports
# ============================================================

def save_feature_filter_qc(datasets):
    """保存被剔除的非 biomarker 特征，方便论文方法部分说明。"""

    rows = []

    for method, (_, X_df, y, sid, drop_cols) in datasets.items():

        for col in drop_cols:
            rows.append({
                "Method": method,
                "DroppedFeature": col,
                "Reason": "algorithm/window/non-biomarker variable",
            })

    qc_df = pd.DataFrame(rows)
    qc_df.to_csv(
        OUT_DIR / "dropped_non_biomarker_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return qc_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Load subject-level feature tables")
    print("=" * 70)

    datasets = {
        "event_guided": load_table(EVENT_FILE),
        "fixed": load_table(FIXED_FILE),
    }

    for method, (_, X_df, y, sid, drop_cols) in datasets.items():
        print(
            f"{method}: X={X_df.shape}, "
            f"label distribution={pd.Series(y).value_counts().to_dict()}, "
            f"dropped non-biomarker features={len(drop_cols)}"
        )

    save_feature_filter_qc(datasets)

    print("\n" + "=" * 70)
    print("1. Feature group ablation")
    print("=" * 70)

    ablation_df = run_feature_group_ablation(datasets)

    print("\nAblation results:")
    print(
        ablation_df.sort_values(
            ["Method", "AUC", "ACC"],
            ascending=[True, False, False],
        )
    )

    print("\n" + "=" * 70)
    print("2. LOSO selection frequency")
    print("=" * 70)

    all_freq = []

    for method, (_, X_df, y, sid, drop_cols) in datasets.items():

        freq_df = loso_selection_frequency(
            X_df=X_df,
            y=y,
            method_name=method,
            k_features=K_IMPORTANCE,
        )

        all_freq.append(freq_df)

    all_freq_df = pd.concat(all_freq, ignore_index=True)
    all_freq_df.to_csv(
        OUT_DIR / "all_loso_selection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stable_df, stable_summary = summarize_stable_features(all_freq_df)

    print("\nStable selected features:")
    print(stable_df.head(30))

    print("\nStable feature group summary:")
    print(stable_summary)

    print("\nFinished.")
    print("Saved to:", OUT_DIR.resolve())
