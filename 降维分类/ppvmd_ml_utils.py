# -*- coding: utf-8 -*-
"""
ppvmd_ml_utils.py

我把多受试者合并、受试者级聚合、特征筛选、降维和机器学习分类函数集中放在这里。
这个文件专门服务于结肠压力非线性特征的 healthy vs STC/patient 分类实验。

重要原则：
1. 我只把 SubjectID 作为独立样本单位。
2. 我不把同一个受试者的多个窗口当成独立样本。
3. 我把 event-guided 和 fixed-window 分开建模，再比较两套方案。
4. 我在交叉验证内部完成标准化、缺失值填充、特征筛选和分类，避免数据泄漏。
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    _HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    _HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    _HAS_CATBOOST = False


# ============================================================
# 1. File discovery and merge
# ============================================================


def find_subject_feature_files(
    root_dir: str | Path,
    event_suffix: str = "_event_guided_phase_features.xlsx",
    fixed_suffix: str = "_fixed_phase_features.xlsx",
    main_subdir: str = "main",
) -> Tuple[List[Path], List[Path]]:
    """我在 results 目录下寻找每个受试者导出的 event 和 fixed 特征表。"""
    root_dir = Path(root_dir)
    event_files: List[Path] = []
    fixed_files: List[Path] = []

    for subject_dir in sorted(root_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        main_dir = subject_dir / main_subdir
        if not main_dir.exists():
            continue

        event_file = main_dir / f"{subject_dir.name}{event_suffix}"
        fixed_file = main_dir / f"{subject_dir.name}{fixed_suffix}"

        if event_file.exists():
            event_files.append(event_file)
        if fixed_file.exists():
            fixed_files.append(fixed_file)

    return event_files, fixed_files


def read_feature_table(path: str | Path) -> pd.DataFrame:
    """我根据文件后缀读取 csv/xlsx 特征表。"""
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def merge_feature_files(files: Sequence[str | Path], method_name: Optional[str] = None) -> pd.DataFrame:
    """我把多个受试者的同类窗口特征表合并成一个窗口级总表。"""
    dfs = []
    for path in files:
        path = Path(path)
        df = read_feature_table(path)

        # 我从文件名兜底提取 SubjectID，避免个别表里 SubjectID 缺失。
        if "SubjectID" not in df.columns:
            subject_id = path.name.split("_event_guided_phase_features")[0]
            subject_id = subject_id.split("_fixed_phase_features")[0]
            df["SubjectID"] = subject_id

        if method_name is not None:
            df["WindowMethod"] = method_name

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def build_event_fixed_window_tables(
    root_dir: str | Path,
    output_dir: str | Path,
    event_suffix: str = "_event_guided_phase_features.xlsx",
    fixed_suffix: str = "_fixed_phase_features.xlsx",
    main_subdir: str = "main",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """我一键合并所有受试者的 event-guided 和 fixed-window 窗口级特征表。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_files, fixed_files = find_subject_feature_files(
        root_dir=root_dir,
        event_suffix=event_suffix,
        fixed_suffix=fixed_suffix,
        main_subdir=main_subdir,
    )

    event_df = merge_feature_files(event_files, method_name="event_guided")
    fixed_df = merge_feature_files(fixed_files, method_name="fixed")

    event_df.to_csv(output_dir / "all_event_window_features.csv", index=False, encoding="utf-8-sig")
    fixed_df.to_csv(output_dir / "all_fixed_window_features.csv", index=False, encoding="utf-8-sig")

    return event_df, fixed_df


# ============================================================
# 2. Subject-level aggregation
# ============================================================


DEFAULT_META_COLS = [
    "SubjectID", "Label", "WindowMethod", "WindowID",
    "StartTime", "EndTime", "CenterTime", "StartIndex", "EndIndex",
    "NumPoints", "EventID", "EventStartTime", "EventEndTime", "EventCoverage",
]


def infer_feature_columns(
    df: pd.DataFrame,
    meta_cols: Sequence[str] = DEFAULT_META_COLS,
    include_prefixes: Optional[Sequence[str]] = None,
    exclude_patterns: Optional[Sequence[str]] = None,
) -> List[str]:
    """我从窗口级表中自动识别可用于聚合和分类的数值特征列。"""
    meta = set(meta_cols)
    feature_cols = []

    for col in df.columns:
        if col in meta:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if include_prefixes is not None and not any(col.startswith(prefix) for prefix in include_prefixes):
            continue
        if exclude_patterns is not None and any(pat in col for pat in exclude_patterns):
            continue
        feature_cols.append(col)

    return feature_cols


def iqr(values: Sequence[float]) -> float:
    """我计算四分位距，用来描述窗口间稳健离散程度。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))


def aggregate_window_to_subject(
    window_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    subject_col: str = "SubjectID",
    label_col: str = "Label",
    aggregations: Sequence[str] = ("mean", "std", "median", "min", "max", "iqr"),
) -> pd.DataFrame:
    """我把窗口级特征聚合为受试者级特征。后续分类只使用这个表。"""
    df = window_df.copy()
    if feature_cols is None:
        feature_cols = infer_feature_columns(df)

    rows = []
    for subject_id, sub in df.groupby(subject_col, dropna=False):
        row = {subject_col: subject_id}
        if label_col in sub.columns:
            row[label_col] = sub[label_col].iloc[0]
        row["NumWindows"] = len(sub)

        for col in feature_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                for agg in aggregations:
                    row[f"{col}__{agg}"] = np.nan
                continue
            if "mean" in aggregations:
                row[f"{col}__mean"] = float(np.mean(vals))
            if "std" in aggregations:
                row[f"{col}__std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if "median" in aggregations:
                row[f"{col}__median"] = float(np.median(vals))
            if "min" in aggregations:
                row[f"{col}__min"] = float(np.min(vals))
            if "max" in aggregations:
                row[f"{col}__max"] = float(np.max(vals))
            if "iqr" in aggregations:
                row[f"{col}__iqr"] = iqr(vals)
        rows.append(row)

    return pd.DataFrame(rows)


def build_subject_level_tables(
    event_window_df: pd.DataFrame,
    fixed_window_df: pd.DataFrame,
    output_dir: str | Path,
    aggregations: Sequence[str] = ("mean", "std", "median", "iqr"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """我分别构建 event-guided 和 fixed-window 的受试者级分类表。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_feature_cols = infer_feature_columns(event_window_df)
    fixed_feature_cols = infer_feature_columns(fixed_window_df)

    common_cols = sorted(set(event_feature_cols).intersection(set(fixed_feature_cols)))

    event_subject_df = aggregate_window_to_subject(
        event_window_df,
        feature_cols=common_cols,
        aggregations=aggregations,
    )
    fixed_subject_df = aggregate_window_to_subject(
        fixed_window_df,
        feature_cols=common_cols,
        aggregations=aggregations,
    )

    event_subject_df.to_csv(output_dir / "all_event_subject_features.csv", index=False, encoding="utf-8-sig")
    fixed_subject_df.to_csv(output_dir / "all_fixed_subject_features.csv", index=False, encoding="utf-8-sig")

    return event_subject_df, fixed_subject_df


# ============================================================
# 3. Data quality checks
# ============================================================


def check_subject_table(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """我检查受试者级表的样本数、标签分布、缺失率和常数特征数量。"""
    feature_cols = [c for c in df.columns if c not in ["SubjectID", label_col, "NumWindows"]]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    constant_cols = []
    for c in numeric_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) > 0 and vals.nunique() <= 1:
            constant_cols.append(c)

    info = {
        "n_subjects": df["SubjectID"].nunique() if "SubjectID" in df.columns else len(df),
        "n_rows": len(df),
        "n_features": len(numeric_cols),
        "n_constant_features": len(constant_cols),
        "max_missing_rate": df[numeric_cols].isna().mean().max() if numeric_cols else np.nan,
        "label_counts": json.dumps(df[label_col].value_counts(dropna=False).to_dict(), ensure_ascii=False) if label_col in df.columns else "{}",
    }
    return pd.DataFrame([info])


# ============================================================
# 4. Feature selection helpers
# ============================================================


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """我计算 Cliff's delta，用非参数效应量表示两组差异方向和强度。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(x) * len(y)))


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """我计算 Cohen's d，用标准化均值差描述效应量。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = ((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1)) / max(len(x)+len(y)-2, 1)
    if pooled <= 1e-12:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """我用 Benjamini-Hochberg 方法做 FDR 校正。"""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q_ranked = ranked * m / (np.arange(m) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q_valid = np.empty_like(q_ranked)
    q_valid[order] = q_ranked
    q[valid] = q_valid
    return q


def univariate_feature_screening(
    subject_df: pd.DataFrame,
    label_col: str = "Label",
    exclude_cols: Sequence[str] = ("SubjectID", "Label", "NumWindows"),
) -> pd.DataFrame:
    """我做单变量统计筛选，用于解释性排序，不直接在全数据上作为最终模型筛选。"""
    feature_cols = [
        c for c in subject_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(subject_df[c])
    ]
    labels = sorted(subject_df[label_col].dropna().unique())
    if len(labels) != 2:
        raise ValueError("Label must contain exactly two classes.")
    a, b = labels[0], labels[1]
    rows = []
    for c in feature_cols:
        x = pd.to_numeric(subject_df.loc[subject_df[label_col] == a, c], errors="coerce").dropna().to_numpy(float)
        y = pd.to_numeric(subject_df.loc[subject_df[label_col] == b, c], errors="coerce").dropna().to_numpy(float)
        if len(x) < 3 or len(y) < 3:
            continue
        try:
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        rows.append({
            "Feature": c,
            "ClassA": a,
            "ClassB": b,
            "N_A": len(x),
            "N_B": len(y),
            "Mean_A": np.mean(x),
            "Mean_B": np.mean(y),
            "Median_A": np.median(x),
            "Median_B": np.median(y),
            "MannWhitney_p": p,
            "CohenD": cohens_d(x, y),
            "CliffsDelta": cliffs_delta(x, y),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["FDR_q"] = bh_fdr(out["MannWhitney_p"])
        out["AbsCliffsDelta"] = out["CliffsDelta"].abs()
        out["AbsCohenD"] = out["CohenD"].abs()
        out = out.sort_values(["AbsCliffsDelta", "AbsCohenD"], ascending=False).reset_index(drop=True)
    return out


# ============================================================
# 5. Model definitions
# ============================================================


def get_classifier_zoo(random_state: int = 42, class_weight: str | None = "balanced") -> Dict[str, object]:
    """我定义一组适合小样本医学分类的模型。"""
    models: Dict[str, object] = {
        "Logistic_L2": LogisticRegression(
            penalty="l2", C=1.0, max_iter=5000, solver="liblinear", class_weight=class_weight, random_state=random_state
        ),
        "Logistic_L1": LogisticRegression(
            penalty="l1", C=0.5, max_iter=5000, solver="liblinear", class_weight=class_weight, random_state=random_state
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight=class_weight, random_state=random_state
        ),
        "SVM_Linear": SVC(
            kernel="linear", C=0.5, probability=True, class_weight=class_weight, random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=2, class_weight=class_weight, random_state=random_state
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=2, class_weight=class_weight, random_state=random_state
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=80, learning_rate=0.05, max_depth=2, random_state=random_state
        ),
        "KNN": KNeighborsClassifier(n_neighbors=3, weights="distance"),
        "GaussianNB": GaussianNB(),
    }

    if _HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )

    if _HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
        )

    if _HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=80,
            depth=2,
            learning_rate=0.05,
            verbose=False,
            random_state=random_state,
        )

    return models


# ============================================================
# 6. Cross-validation and metrics
# ============================================================


def make_cv(y: Sequence[int], random_state: int = 42, preferred_splits: int = 5):
    """我根据小样本标签分布自动选择合适的交叉验证策略。"""
    y = np.asarray(y)
    class_counts = pd.Series(y).value_counts()
    min_count = int(class_counts.min())

    if min_count < 2:
        raise ValueError("Each class needs at least 2 subjects for cross-validation.")

    n_splits = min(preferred_splits, min_count)

    if n_splits >= 3:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return LeaveOneOut()


def calculate_binary_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    """我统一计算医学二分类常用指标。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = np.nan

    sen = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spe = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = np.nan

    return {
        "ACC": accuracy_score(y_true, y_pred),
        "Balanced_ACC": balanced_accuracy_score(y_true, y_pred),
        "SEN": sen,
        "SPE": spe,
        "PPV": ppv,
        "NPV": npv,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def get_model_score(model, X_test):
    """我尽量从模型中取出正类概率或连续得分，用于 AUC。"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return model.predict(X_test)


def build_model_pipeline(
    classifier,
    scaler: str = "standard",
    feature_selection: str = "kbest_f",
    k_features: int = 20,
    pca_components: Optional[int] = None,
    random_state: int = 42,
) -> Pipeline:
    """我构建完整 pipeline，确保所有预处理都在交叉验证内部完成。"""
    steps = []
    steps.append(("imputer", SimpleImputer(strategy="median")))

    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler == "none":
        pass
    else:
        raise ValueError(f"Unknown scaler: {scaler}")

    if feature_selection == "kbest_f":
        steps.append(("select", SelectKBest(score_func=f_classif, k=k_features)))
    elif feature_selection == "kbest_mi":
        def mi_func(X, y):
            return mutual_info_classif(X, y, random_state=random_state)
        steps.append(("select", SelectKBest(score_func=mi_func, k=k_features)))
    elif feature_selection == "none":
        pass
    else:
        raise ValueError(f"Unknown feature_selection: {feature_selection}")

    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=random_state)))

    steps.append(("clf", classifier))
    return Pipeline(steps)


def evaluate_models_cv(
    subject_df: pd.DataFrame,
    label_col: str = "Label",
    subject_col: str = "SubjectID",
    feature_cols: Optional[Sequence[str]] = None,
    models: Optional[Dict[str, object]] = None,
    random_state: int = 42,
    preferred_splits: int = 5,
    k_features: int = 20,
    scaler: str = "standard",
    feature_selection: str = "kbest_f",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """我用交叉验证评估多个分类器，并输出总体指标和逐折预测。"""
    df = subject_df.copy()
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in [subject_col, label_col, "NumWindows"] and pd.api.types.is_numeric_dtype(df[c])
        ]

    X = df[list(feature_cols)].copy()
    y = df[label_col].astype(int).to_numpy()
    subjects = df[subject_col].astype(str).to_numpy() if subject_col in df.columns else np.arange(len(df)).astype(str)

    # 我避免 k 大于当前特征数。
    k_use = min(k_features, X.shape[1])

    if models is None:
        models = get_classifier_zoo(random_state=random_state)

    cv = make_cv(y, random_state=random_state, preferred_splits=preferred_splits)

    summary_rows = []
    pred_rows = []

    for model_name, clf in models.items():
        y_true_all = []
        y_pred_all = []
        y_score_all = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y) if not isinstance(cv, LeaveOneOut) else cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 我跳过训练集中只有一个类别的异常折。
            if len(np.unique(y_train)) < 2:
                continue

            pipe = build_model_pipeline(
                classifier=clone(clf),
                scaler=scaler,
                feature_selection=feature_selection,
                k_features=k_use,
                random_state=random_state,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_score = get_model_score(pipe, X_test)

            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(np.asarray(y_pred).tolist())
            y_score_all.extend(np.asarray(y_score).ravel().tolist())

            for s, yt, yp, ys in zip(subjects[test_idx], y_test, y_pred, np.asarray(y_score).ravel()):
                pred_rows.append({
                    "Model": model_name,
                    "Fold": fold_idx,
                    "SubjectID": s,
                    "TrueLabel": int(yt),
                    "PredLabel": int(yp),
                    "Score": float(ys),
                })

        metrics = calculate_binary_metrics(y_true_all, y_pred_all, y_score_all)
        metrics["Model"] = model_name
        metrics["N_Subjects"] = len(df)
        metrics["N_Features_Total"] = X.shape[1]
        metrics["N_Features_Selected"] = k_use
        metrics["Scaler"] = scaler
        metrics["FeatureSelection"] = feature_selection
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df):
        summary_df = summary_df.sort_values(["AUC", "Balanced_ACC", "F1"], ascending=False).reset_index(drop=True)
    pred_df = pd.DataFrame(pred_rows)
    return summary_df, pred_df


# ============================================================
# 7. Event vs fixed experiment
# ============================================================


def run_event_fixed_ml_experiment(
    event_subject_df: pd.DataFrame,
    fixed_subject_df: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
    preferred_splits: int = 5,
    k_features_list: Sequence[int] = (10, 20, 30),
    scaler: str = "standard",
    feature_selection: str = "kbest_f",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """我分别对 event-guided 与 fixed-window 受试者级表建模，并比较分类性能。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_preds = []

    for method_name, df in [("event_guided", event_subject_df), ("fixed", fixed_subject_df)]:
        for k in k_features_list:
            summary, preds = evaluate_models_cv(
                df,
                random_state=random_state,
                preferred_splits=preferred_splits,
                k_features=k,
                scaler=scaler,
                feature_selection=feature_selection,
            )
            summary.insert(0, "Method", method_name)
            preds.insert(0, "Method", method_name)
            all_summary.append(summary)
            all_preds.append(preds)

    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    summary_df.to_csv(output_dir / "event_vs_fixed_ml_summary.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(output_dir / "event_vs_fixed_ml_predictions.csv", index=False, encoding="utf-8-sig")

    return summary_df, pred_df
