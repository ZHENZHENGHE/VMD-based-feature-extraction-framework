# -*- coding: utf-8 -*-
"""
ppvmd_feature_classification.py

Feature selection and classification module for physiology-preserving VMD-assisted
phase-space/RQA features.

中文说明：
    本文件用于“特征降维 + 多分类器比较”。建议作为独立脚本使用，不要放进
    预处理/VMD/相空间重构脚本中，以保证SCI论文中的方法模块清晰可复现。

Core design for SCI-level rigor:
    1. Subject-wise validation / 按受试者验证，避免同一受试者窗口泄漏到训练和测试。
    2. Feature selection inside each CV fold / 每个交叉验证fold内部做特征筛选，避免信息泄漏。
    3. Missing-value, variance, correlation, and effect-size filtering / 多层特征筛选。
    4. Multiple conventional classifiers / 多个传统分类器对比，避免深度学习小样本过拟合。
    5. Selected-feature frequency / 统计特征被选中的稳定性。

Expected input:
    A CSV or DataFrame containing either:
        A) subject-level features: one row per subject per method
        B) window-level features: multiple rows per subject; use aggregate_to_subject_level first

Essential columns:
    - SubjectID
    - Label: 0 = healthy, 1 = patient/STC
    - WindowMethod: event_guided or fixed, optional but recommended

Author: generated for PPVMD workflow
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# 0. Utility functions / 基础工具函数
# ============================================================

META_COLUMNS_DEFAULT = {
    "SubjectID", "Label", "WindowMethod", "WindowID", "StartTime", "EndTime",
    "CenterTime", "StartIndex", "EndIndex", "NumPoints", "EventID",
    "EventStartTime", "EventEndTime", "WindowCoverage", "EventCoverage",
}


def safe_numeric_frame(df: pd.DataFrame, exclude_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Keep numeric feature columns only.

    中文：只保留数值型特征列，自动排除SubjectID、Label等元数据列。
    """
    exclude = set(META_COLUMNS_DEFAULT)
    if exclude_cols is not None:
        exclude |= set(exclude_cols)

    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.select_dtypes(include=[np.number])
    return X


def cliff_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Cliff's delta effect size.

    中文：Cliff's delta（非参数效应量），范围[-1,1]。
    绝对值越大，组间差异越强。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Cohen's d effect size.

    中文：Cohen's d（均值差异效应量）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2) + 1e-12)
    return float((np.mean(x) - np.mean(y)) / pooled)


# ============================================================
# 1. Subject-level aggregation / 受试者级聚合
# ============================================================

def aggregate_to_subject_level(
    window_feature_df: pd.DataFrame,
    subject_col: str = "SubjectID",
    label_col: str = "Label",
    method_col: str = "WindowMethod",
    aggregations: Sequence[str] = ("mean", "std", "median", "min", "max"),
    exclude_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate window-level features to subject-level features.

    中文：将窗口级特征聚合为受试者级特征。
    这是避免pseudo-replication（伪重复）的关键步骤。

    Output:
        One row per SubjectID + WindowMethod.
    """
    df = window_feature_df.copy()
    if subject_col not in df.columns:
        raise ValueError(f"Missing required column: {subject_col}")
    if label_col not in df.columns:
        raise ValueError(f"Missing required column: {label_col}")
    if method_col not in df.columns:
        df[method_col] = "unknown"

    X_num = safe_numeric_frame(df, exclude_cols=exclude_cols)
    meta = df[[subject_col, label_col, method_col]].copy()
    tmp = pd.concat([meta, X_num], axis=1)

    group_cols = [subject_col, label_col, method_col]
    agg = tmp.groupby(group_cols, dropna=False).agg(aggregations)
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    # Add number of windows per subject/method.
    nwin = tmp.groupby(group_cols, dropna=False).size().reset_index(name="NumWindows")
    out = agg.merge(nwin, on=group_cols, how="left")
    return out


# ============================================================
# 2. sklearn-compatible feature selectors / 可放入Pipeline的特征筛选器
# ============================================================

class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Median imputation while preserving pandas DataFrame columns.

    中文：中位数填补缺失值，并保留特征名。
    """
    def __init__(self, strategy: str = "median"):
        self.strategy = strategy

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        if self.strategy == "median":
            self.fill_values_ = X.median(numeric_only=True)
        elif self.strategy == "mean":
            self.fill_values_ = X.mean(numeric_only=True)
        else:
            raise ValueError("Only median or mean strategy is supported.")
        self.fill_values_ = self.fill_values_.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.columns = self.columns_[: X.shape[1]]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.fill_values_)
        return X


class MissingVarianceFilter(BaseEstimator, TransformerMixin):
    """
    Remove features with high missing rate or near-zero variance.

    中文：删除缺失率过高或几乎无变化的特征。
    """
    def __init__(self, max_missing_rate: float = 0.30, min_variance: float = 1e-10):
        self.max_missing_rate = max_missing_rate
        self.min_variance = min_variance

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.input_features_ = list(X.columns)
        missing_rate = X.isna().mean()
        variance = X.var(axis=0, skipna=True)
        keep = (missing_rate <= self.max_missing_rate) & (variance > self.min_variance)
        self.keep_features_ = list(keep[keep].index)
        if len(self.keep_features_) == 0:
            # fallback: keep all columns to avoid crashing
            self.keep_features_ = self.input_features_
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.columns = self.input_features_[: X.shape[1]]
        return X[self.keep_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_features_)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Remove highly correlated features.

    中文：删除高度相关特征，降低多重共线性。
    """
    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.input_features_ = list(X.columns)
        if X.shape[1] <= 1:
            self.keep_features_ = self.input_features_
            return self

        corr = X.corr(method="spearman").abs().fillna(0.0)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = set()
        for col in upper.columns:
            if any(upper[col] > self.threshold):
                drop.add(col)
        self.keep_features_ = [c for c in self.input_features_ if c not in drop]
        if len(self.keep_features_) == 0:
            self.keep_features_ = self.input_features_[:1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.columns = self.input_features_[: X.shape[1]]
        return X[self.keep_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_features_)


class EffectSizeSelector(BaseEstimator, TransformerMixin):
    """
    Supervised feature selector based on effect size computed on training fold only.

    中文：基于训练集内效应量筛选特征，避免信息泄漏。
    推荐小样本使用Cliff's delta或Cohen's d排序，而不是过度依赖p值。
    """
    def __init__(
        self,
        k: int = 15,
        score_method: str = "cliff_abs",
        min_abs_score: Optional[float] = None,
    ):
        self.k = k
        self.score_method = score_method
        self.min_abs_score = min_abs_score

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y)
        self.input_features_ = list(X.columns)

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("EffectSizeSelector currently supports binary labels only.")
        c0, c1 = classes[0], classes[1]

        rows = []
        for col in self.input_features_:
            x0 = X.loc[y == c0, col].to_numpy(dtype=float)
            x1 = X.loc[y == c1, col].to_numpy(dtype=float)
            cd = cliff_delta(x1, x0)  # positive means class1 > class0
            d = cohens_d(x1, x0)
            try:
                p = stats.mannwhitneyu(
                    x1[np.isfinite(x1)],
                    x0[np.isfinite(x0)],
                    alternative="two-sided"
                ).pvalue
            except Exception:
                p = np.nan
            if self.score_method == "cliff_abs":
                score = abs(cd) if np.isfinite(cd) else 0.0
            elif self.score_method == "cohen_abs":
                score = abs(d) if np.isfinite(d) else 0.0
            elif self.score_method == "mw_neglogp":
                score = -np.log10(p + 1e-12) if np.isfinite(p) else 0.0
            else:
                raise ValueError("score_method must be cliff_abs, cohen_abs, or mw_neglogp")
            rows.append({
                "Feature": col,
                "Score": score,
                "CliffDelta": cd,
                "CohensD": d,
                "MannWhitneyP": p,
            })

        table = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

        if self.min_abs_score is not None:
            table = table[table["Score"] >= self.min_abs_score].copy()

        if len(table) == 0:
            table = pd.DataFrame(rows).sort_values("Score", ascending=False).head(1)

        self.score_table_ = table.copy()
        self.selected_features_ = table.head(self.k)["Feature"].tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.columns = self.input_features_[: X.shape[1]]
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_)


# ============================================================
# 3. Classifier definitions / 分类模型定义
# ============================================================

def get_default_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Default small-sample classifiers.

    中文：适合小样本医学数据的传统分类器。
    """
    return {
        "LogisticRegression": LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=500,
            random_state=random_state,
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            C=1.0,
            gamma="scale",
            random_state=random_state,
        ),
        "SVM_Linear": SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            C=1.0,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=3,
        ),
    }


# ============================================================
# 4. CV evaluation / 交叉验证评估
# ============================================================

@dataclass
class ClassificationResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    selected_feature_frequency: pd.DataFrame
    fold_selected_features: pd.DataFrame


def _get_positive_scores(model, X_test) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X_test)
        return np.asarray(s, dtype=float)
    pred = model.predict(X_test)
    return np.asarray(pred, dtype=float)


def _compute_binary_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity_Recall": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["Specificity"] = tn / (tn + fp + 1e-12)
        out["TP"] = int(tp)
        out["TN"] = int(tn)
        out["FP"] = int(fp)
        out["FN"] = int(fn)
    except Exception:
        out["Specificity"] = np.nan
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["ROC_AUC"] = roc_auc_score(y_true, y_score)
        except Exception:
            out["ROC_AUC"] = np.nan
    else:
        out["ROC_AUC"] = np.nan
    return out


def build_selection_pipeline(
    classifier,
    max_missing_rate: float = 0.30,
    min_variance: float = 1e-10,
    corr_threshold: float = 0.90,
    top_k: int = 15,
    effect_method: str = "cliff_abs",
) -> Pipeline:
    """
    Build a leakage-safe feature-selection + classifier pipeline.

    中文：构建“防信息泄漏”的特征筛选和分类管道。
    注意：所有筛选器都在训练fold内部fit。
    """
    return Pipeline([
        ("missing_variance", MissingVarianceFilter(max_missing_rate=max_missing_rate, min_variance=min_variance)),
        ("imputer", DataFrameImputer(strategy="median")),
        ("correlation", CorrelationFilter(threshold=corr_threshold)),
        ("effect_size", EffectSizeSelector(k=top_k, score_method=effect_method)),
        ("scaler", StandardScaler()),
        ("clf", classifier),
    ])


def evaluate_subject_level_classification(
    subject_feature_df: pd.DataFrame,
    label_col: str = "Label",
    subject_col: str = "SubjectID",
    method_col: Optional[str] = "WindowMethod",
    target_method: Optional[str] = None,
    exclude_cols: Optional[Iterable[str]] = None,
    top_k: int = 15,
    corr_threshold: float = 0.90,
    max_missing_rate: float = 0.30,
    random_state: int = 42,
    classifiers: Optional[Dict[str, Any]] = None,
) -> ClassificationResult:
    """
    Subject-level classification with Leave-One-Subject-Out style evaluation.

    中文：受试者级分类。每一行代表一个受试者，使用Leave-One-Out（留一法）。
    适用于已经完成subject-level aggregation的数据表。
    """
    df = subject_feature_df.copy()
    if target_method is not None and method_col is not None and method_col in df.columns:
        df = df[df[method_col] == target_method].copy()

    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)

    if df[subject_col].nunique() < 4:
        warnings.warn("Very few subjects. Results are exploratory only.")
    if df[label_col].nunique() < 2:
        raise ValueError("Label column must contain two classes.")

    X = safe_numeric_frame(df, exclude_cols=exclude_cols)
    y = df[label_col].to_numpy(dtype=int)
    subjects = df[subject_col].astype(str).to_numpy()

    if classifiers is None:
        classifiers = get_default_classifiers(random_state=random_state)

    loo = LeaveOneOut()
    pred_rows = []
    feature_rows = []

    for model_name, clf in classifiers.items():
        fold_id = 0
        for train_idx, test_idx in loo.split(X, y):
            fold_id += 1
            pipe = build_selection_pipeline(
                clone(clf),
                max_missing_rate=max_missing_rate,
                corr_threshold=corr_threshold,
                top_k=top_k,
            )
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y[train_idx]
            y_test = y[test_idx]

            # If a training fold has only one class, skip.
            if len(np.unique(y_train)) < 2:
                continue

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_score = _get_positive_scores(pipe, X_test)

            # selected features in this fold
            try:
                selected = pipe.named_steps["effect_size"].selected_features_
            except Exception:
                selected = []
            for f in selected:
                feature_rows.append({
                    "Model": model_name,
                    "Fold": fold_id,
                    "Feature": f,
                })

            pred_rows.append({
                "Model": model_name,
                "Fold": fold_id,
                "SubjectID": subjects[test_idx[0]],
                "TrueLabel": int(y_test[0]),
                "PredLabel": int(y_pred[0]),
                "Score": float(y_score[0]) if len(np.ravel(y_score)) else np.nan,
            })

    pred_df = pd.DataFrame(pred_rows)
    if len(pred_df) == 0:
        raise RuntimeError("No predictions generated. Check label distribution and data size.")

    summary_rows = []
    for model_name, g in pred_df.groupby("Model"):
        metrics = _compute_binary_metrics(
            g["TrueLabel"].to_numpy(),
            g["PredLabel"].to_numpy(),
            g["Score"].to_numpy(),
        )
        metrics["Model"] = model_name
        metrics["N_Subjects"] = len(g)
        summary_rows.append(metrics)
    summary = pd.DataFrame(summary_rows).sort_values("BalancedAccuracy", ascending=False).reset_index(drop=True)

    fold_selected = pd.DataFrame(feature_rows)
    if len(fold_selected):
        freq = (
            fold_selected.groupby(["Model", "Feature"])
            .size()
            .reset_index(name="SelectedCount")
        )
        total_folds = pred_df.groupby("Model")["Fold"].nunique().reset_index(name="TotalFolds")
        freq = freq.merge(total_folds, on="Model", how="left")
        freq["SelectionFrequency"] = freq["SelectedCount"] / freq["TotalFolds"]
        freq = freq.sort_values(["Model", "SelectionFrequency"], ascending=[True, False]).reset_index(drop=True)
    else:
        freq = pd.DataFrame(columns=["Model", "Feature", "SelectedCount", "TotalFolds", "SelectionFrequency"])

    return ClassificationResult(
        predictions=pred_df,
        summary=summary,
        selected_feature_frequency=freq,
        fold_selected_features=fold_selected,
    )


def evaluate_window_level_subjectwise_classification(
    window_feature_df: pd.DataFrame,
    label_col: str = "Label",
    subject_col: str = "SubjectID",
    method_col: Optional[str] = "WindowMethod",
    target_method: Optional[str] = None,
    exclude_cols: Optional[Iterable[str]] = None,
    top_k: int = 15,
    corr_threshold: float = 0.90,
    max_missing_rate: float = 0.30,
    random_state: int = 42,
    classifiers: Optional[Dict[str, Any]] = None,
) -> ClassificationResult:
    """
    Window-level classification with Leave-One-Subject-Out validation.

    中文：窗口级样本分类，但按受试者分组验证。
    关键：同一受试者的所有窗口必须同时在训练集或测试集，避免数据泄漏。
    """
    df = window_feature_df.copy()
    if target_method is not None and method_col is not None and method_col in df.columns:
        df = df[df[method_col] == target_method].copy()

    df = df.dropna(subset=[label_col, subject_col]).copy()
    df[label_col] = df[label_col].astype(int)

    X = safe_numeric_frame(df, exclude_cols=exclude_cols)
    y = df[label_col].to_numpy(dtype=int)
    groups = df[subject_col].astype(str).to_numpy()

    if classifiers is None:
        classifiers = get_default_classifiers(random_state=random_state)

    logo = LeaveOneGroupOut()
    pred_rows = []
    feature_rows = []

    for model_name, clf in classifiers.items():
        fold_id = 0
        for train_idx, test_idx in logo.split(X, y, groups=groups):
            fold_id += 1
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                continue

            pipe = build_selection_pipeline(
                clone(clf),
                max_missing_rate=max_missing_rate,
                corr_threshold=corr_threshold,
                top_k=top_k,
            )
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_score = _get_positive_scores(pipe, X_test)

            try:
                selected = pipe.named_steps["effect_size"].selected_features_
            except Exception:
                selected = []
            for f in selected:
                feature_rows.append({"Model": model_name, "Fold": fold_id, "Feature": f})

            for local_i, idx in enumerate(test_idx):
                pred_rows.append({
                    "Model": model_name,
                    "Fold": fold_id,
                    "SubjectID": str(groups[idx]),
                    "WindowID": df.iloc[idx].get("WindowID", np.nan),
                    "TrueLabel": int(y[idx]),
                    "PredLabel": int(y_pred[local_i]),
                    "Score": float(y_score[local_i]) if len(np.ravel(y_score)) > local_i else np.nan,
                })

    pred_df = pd.DataFrame(pred_rows)
    if len(pred_df) == 0:
        raise RuntimeError("No predictions generated. Check label distribution and group sizes.")

    # Window-level metrics
    summary_rows = []
    for model_name, g in pred_df.groupby("Model"):
        metrics = _compute_binary_metrics(
            g["TrueLabel"].to_numpy(),
            g["PredLabel"].to_numpy(),
            g["Score"].to_numpy(),
        )
        metrics["Model"] = model_name
        metrics["N_Windows"] = len(g)
        metrics["N_Subjects"] = g["SubjectID"].nunique()
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows).sort_values("BalancedAccuracy", ascending=False).reset_index(drop=True)

    fold_selected = pd.DataFrame(feature_rows)
    if len(fold_selected):
        freq = fold_selected.groupby(["Model", "Feature"]).size().reset_index(name="SelectedCount")
        total_folds = pred_df.groupby("Model")["Fold"].nunique().reset_index(name="TotalFolds")
        freq = freq.merge(total_folds, on="Model", how="left")
        freq["SelectionFrequency"] = freq["SelectedCount"] / freq["TotalFolds"]
        freq = freq.sort_values(["Model", "SelectionFrequency"], ascending=[True, False]).reset_index(drop=True)
    else:
        freq = pd.DataFrame(columns=["Model", "Feature", "SelectedCount", "TotalFolds", "SelectionFrequency"])

    return ClassificationResult(
        predictions=pred_df,
        summary=summary,
        selected_feature_frequency=freq,
        fold_selected_features=fold_selected,
    )


# ============================================================
# 5. Convenience workflow / 便捷工作流
# ============================================================

def run_full_feature_selection_classification_workflow(
    feature_df: pd.DataFrame,
    level: str = "subject",
    label_col: str = "Label",
    subject_col: str = "SubjectID",
    method_col: str = "WindowMethod",
    target_method: Optional[str] = "event_guided",
    top_k: int = 15,
    output_prefix: Optional[str] = None,
) -> ClassificationResult:
    """
    One-call workflow.

    中文：一键运行特征筛选和分类。

    level:
        "subject" = 输入已经是受试者级特征
        "window"  = 输入是窗口级特征，按受试者分组验证
    """
    if level == "subject":
        result = evaluate_subject_level_classification(
            feature_df,
            label_col=label_col,
            subject_col=subject_col,
            method_col=method_col,
            target_method=target_method,
            top_k=top_k,
        )
    elif level == "window":
        result = evaluate_window_level_subjectwise_classification(
            feature_df,
            label_col=label_col,
            subject_col=subject_col,
            method_col=method_col,
            target_method=target_method,
            top_k=top_k,
        )
    else:
        raise ValueError("level must be 'subject' or 'window'")

    if output_prefix is not None:
        result.predictions.to_csv(f"{output_prefix}_predictions.csv", index=False)
        result.summary.to_csv(f"{output_prefix}_summary.csv", index=False)
        result.selected_feature_frequency.to_csv(f"{output_prefix}_selected_feature_frequency.csv", index=False)
        result.fold_selected_features.to_csv(f"{output_prefix}_fold_selected_features.csv", index=False)

    return result
