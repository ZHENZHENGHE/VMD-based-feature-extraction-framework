# -*- coding: utf-8 -*-
"""
ppvmd_phase_space_stats.py

Quantitative comparison utilities for physiology-guided phase-space analysis.

中文说明：
    本模块用于比较 event-guided windows 与 fixed 1024-point windows 的
    非线性动力学特征差异，服务于论文结果部分和后续分类前的统计验证。

核心功能：
    1. 自动筛选相空间 / 熵 / RQA / LLE 数值特征；
    2. 对 event-guided vs fixed 进行统计比较；
    3. 计算效应量，包括 Cliff's delta 和 Cohen's d；
    4. 生成适合论文展示的 summary table；
    5. 生成箱线图 / 点图；
    6. 支持窗口级比较和受试者级聚合。

科学注意：
    - 单个受试者内部的窗口并不完全独立，因此窗口级统计只适合作为 exploratory analysis。
    - 真正用于患者 vs 健康分类时，建议先按 SubjectID 聚合为 subject-level feature table，
      然后做 leave-one-subject-out 或 subject-level cross-validation。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, ttest_ind, rankdata


# ============================================================
# 1. Feature selection and cleaning
# ============================================================


def get_default_nonlinear_feature_suffixes() -> Tuple[str, ...]:
    """
    返回默认用于 event-guided vs fixed 比较的非线性特征后缀。
    """
    return (
        "SampEn",
        "PermEn",
        "PS_TrajectoryLength",
        "PS_MeanStep",
        "PS_StepStd",
        "PS_StateSpread",
        "PS_Eig1",
        "PS_Eig2",
        "PS_EigRatio12",
        "PS_LogVolume",
        "RQA_RR",
        "RQA_DET",
        "RQA_LAM",
        "RQA_L_mean",
        "RQA_L_max",
        "RQA_V_mean",
        "RQA_TT",
        "RQA_epsilon",
        "ApproxLLE",
    )


def infer_nonlinear_feature_columns(
    df: pd.DataFrame,
    signal_prefixes: Optional[Sequence[str]] = None,
    suffixes: Optional[Sequence[str]] = None,
    require_numeric: bool = True,
) -> List[str]:
    """
    自动从 feature table 中识别非线性动力学特征列。

    参数：
        signal_prefixes:
            例如 ["Pressure_Clean", "VMD_Reconstructed", "VMD_Mode3", "VMD_Mode4"]。
            None 时不限制前缀。
        suffixes:
            例如 ["SampEn", "RQA_DET", "ApproxLLE"]。
            None 时使用默认后缀。
    """
    if suffixes is None:
        suffixes = get_default_nonlinear_feature_suffixes()

    cols = []
    for col in df.columns:
        if signal_prefixes is not None:
            if not any(col.startswith(prefix + "_") for prefix in signal_prefixes):
                continue
        if any(col.endswith("_" + s) or col.endswith(s) for s in suffixes):
            if require_numeric:
                values = pd.to_numeric(df[col], errors="coerce")
                if values.notna().sum() == 0:
                    continue
            cols.append(col)

    return cols


def coerce_numeric_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """
    将指定特征列安全转换为 numeric。
    """
    out = df.copy()
    for col in feature_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ============================================================
# 2. Effect sizes and statistical tests
# ============================================================


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d: 两组均值差异 / 合并标准差。

    解释：
        |d| ≈ 0.2 小效应；0.5 中等效应；0.8 大效应。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled = ((nx - 1) * sx + (ny - 1) * sy) / max(nx + ny - 2, 1)
    if pooled <= 1e-12:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: 非参数效应量，范围 [-1, 1]。

    delta > 0 表示 x 组整体大于 y 组。
    delta < 0 表示 x 组整体小于 y 组。

    经验解释：
        |delta| < 0.147 negligible
        0.147–0.33 small
        0.33–0.474 medium
        >0.474 large
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan

    # 向量化计算，窗口数通常不大，足够快。
    diff = x[:, None] - y[None, :]
    n_greater = np.sum(diff > 0)
    n_less = np.sum(diff < 0)
    return float((n_greater - n_less) / (len(x) * len(y)))


def benjamini_hochberg_fdr(p_values: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg FDR 校正。
    返回 q-values。
    """
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


def compare_two_window_methods(
    feature_df: pd.DataFrame,
    feature_cols: Sequence[str],
    method_col: str = "WindowMethod",
    method_a: str = "event_guided",
    method_b: str = "fixed",
    min_n_per_group: int = 3,
) -> pd.DataFrame:
    """
    比较两种窗口方法在多个非线性特征上的差异。

    默认比较：event_guided vs fixed。

    返回表格字段：
        Feature
        N_A / N_B
        Mean_A / Mean_B
        Median_A / Median_B
        Std_A / Std_B
        MeanDiff_A_minus_B
        CohenD
        CliffsDelta
        MannWhitney_p
        TTest_p
        FDR_q
    """
    df = feature_df.copy()
    if method_col not in df.columns:
        raise ValueError(f"feature_df 中缺少 {method_col} 列。")

    df = coerce_numeric_features(df, feature_cols)
    rows = []

    for feature in feature_cols:
        if feature not in df.columns:
            continue
        xa = df.loc[df[method_col] == method_a, feature].to_numpy(dtype=float)
        xb = df.loc[df[method_col] == method_b, feature].to_numpy(dtype=float)
        xa = xa[np.isfinite(xa)]
        xb = xb[np.isfinite(xb)]

        row = {
            "Feature": feature,
            "MethodA": method_a,
            "MethodB": method_b,
            "N_A": len(xa),
            "N_B": len(xb),
            "Mean_A": np.nanmean(xa) if len(xa) else np.nan,
            "Mean_B": np.nanmean(xb) if len(xb) else np.nan,
            "Median_A": np.nanmedian(xa) if len(xa) else np.nan,
            "Median_B": np.nanmedian(xb) if len(xb) else np.nan,
            "Std_A": np.nanstd(xa, ddof=1) if len(xa) > 1 else np.nan,
            "Std_B": np.nanstd(xb, ddof=1) if len(xb) > 1 else np.nan,
            "MeanDiff_A_minus_B": np.nan,
            "CohenD": np.nan,
            "CliffsDelta": np.nan,
            "MannWhitney_p": np.nan,
            "TTest_p": np.nan,
        }

        if len(xa) >= min_n_per_group and len(xb) >= min_n_per_group:
            row["MeanDiff_A_minus_B"] = row["Mean_A"] - row["Mean_B"]
            row["CohenD"] = cohens_d(xa, xb)
            row["CliffsDelta"] = cliffs_delta(xa, xb)
            try:
                row["MannWhitney_p"] = float(mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
            except Exception:
                row["MannWhitney_p"] = np.nan
            try:
                row["TTest_p"] = float(ttest_ind(xa, xb, equal_var=False, nan_policy="omit").pvalue)
            except Exception:
                row["TTest_p"] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)
    if len(result):
        result["FDR_q"] = benjamini_hochberg_fdr(result["MannWhitney_p"].to_numpy(dtype=float))
        result["AbsCliffsDelta"] = result["CliffsDelta"].abs()
        result["AbsCohenD"] = result["CohenD"].abs()
        result = result.sort_values(["AbsCliffsDelta", "AbsCohenD"], ascending=False).reset_index(drop=True)
    return result


# ============================================================
# 3. Subject-level aggregation
# ============================================================


def aggregate_features_by_subject(
    feature_df: pd.DataFrame,
    feature_cols: Sequence[str],
    subject_col: str = "SubjectID",
    label_col: str = "Label",
    method_col: str = "WindowMethod",
    aggregations: Sequence[str] = ("mean", "std", "median", "min", "max"),
) -> pd.DataFrame:
    """
    将窗口级特征聚合为受试者级特征。

    为什么重要：
        你的样本数约 20 个受试者。窗口不是独立受试者，不能把所有窗口直接当作完全独立样本。
        用于最终患者 vs 健康分类时，推荐使用 subject-level 聚合表。
    """
    df = feature_df.copy()
    df = coerce_numeric_features(df, feature_cols)

    group_cols = [subject_col]
    if label_col in df.columns:
        group_cols.append(label_col)
    if method_col in df.columns:
        group_cols.append(method_col)

    grouped = df.groupby(group_cols, dropna=False)[list(feature_cols)].agg(list(aggregations))
    grouped.columns = [f"{feat}__{agg}" for feat, agg in grouped.columns]
    out = grouped.reset_index()

    # 加入窗口数量作为质量控制变量。
    counts = df.groupby(group_cols, dropna=False).size().reset_index(name="NumWindows")
    out = out.merge(counts, on=group_cols, how="left")
    return out


# ============================================================
# 4. Plotting utilities
# ============================================================


def plot_method_boxplots(
    feature_df: pd.DataFrame,
    features: Sequence[str],
    method_col: str = "WindowMethod",
    method_order: Sequence[str] = ("event_guided", "fixed"),
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5.0, 4.0),
    show_points: bool = True,
    title: str = "Event-guided vs fixed-window nonlinear features",
):
    """
    为多个特征绘制 event-guided vs fixed 的箱线图。
    不依赖 seaborn，便于在 notebook 中直接运行。
    """
    features = [f for f in features if f in feature_df.columns]
    if len(features) == 0:
        raise ValueError("没有可绘制的 features。")

    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    rng = np.random.default_rng(42)
    for i, feature in enumerate(features):
        ax = axes[i // ncols][i % ncols]
        data = []
        labels = []
        for method in method_order:
            values = pd.to_numeric(
                feature_df.loc[feature_df[method_col] == method, feature],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            data.append(values)
            labels.append(method)

        ax.boxplot(data, labels=labels, showfliers=True)

        if show_points:
            for j, values in enumerate(data, start=1):
                if len(values) == 0:
                    continue
                jitter = rng.normal(0, 0.035, size=len(values))
                ax.scatter(np.full(len(values), j) + jitter, values, s=18, alpha=0.7)

        ax.set_title(feature, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)

    # 删除空轴。
    for j in range(len(features), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    return fig


def plot_top_method_differences(
    comparison_df: pd.DataFrame,
    top_n: int = 20,
    effect_col: str = "CliffsDelta",
    title: str = "Top nonlinear feature differences: event-guided vs fixed",
):
    """
    根据效应量绘制排序条形图。
    """
    if comparison_df.empty:
        raise ValueError("comparison_df 为空。")
    if effect_col not in comparison_df.columns:
        raise ValueError(f"comparison_df 中缺少 {effect_col} 列。")

    plot_df = comparison_df.copy()
    plot_df = plot_df[np.isfinite(plot_df[effect_col])]
    plot_df = plot_df.reindex(plot_df[effect_col].abs().sort_values(ascending=False).index).head(top_n)
    plot_df = plot_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["Feature"], plot_df[effect_col])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(effect_col)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ============================================================
# 5. Convenience report
# ============================================================


def build_event_vs_fixed_report(
    event_feature_df: pd.DataFrame,
    fixed_feature_df: pd.DataFrame,
    signal_prefixes: Optional[Sequence[str]] = ("Pressure_Clean", "VMD_Reconstructed", "VMD_Mode3", "VMD_Mode4", "VMD_Mode3_plus_Mode4"),
    selected_suffixes: Optional[Sequence[str]] = None,
    min_n_per_group: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    一键构建 event-guided vs fixed-window 的定量比较报告。

    返回：
        {
            "combined": 合并后的窗口级特征表,
            "feature_columns": 特征列清单表,
            "comparison": 统计比较结果,
            "subject_level": 受试者级聚合表
        }
    """
    e = event_feature_df.copy()
    f = fixed_feature_df.copy()
    e["WindowMethod"] = "event_guided"
    f["WindowMethod"] = "fixed"
    combined = pd.concat([e, f], ignore_index=True)

    feature_cols = infer_nonlinear_feature_columns(
        combined,
        signal_prefixes=signal_prefixes,
        suffixes=selected_suffixes,
        require_numeric=True,
    )

    comparison = compare_two_window_methods(
        combined,
        feature_cols=feature_cols,
        method_col="WindowMethod",
        method_a="event_guided",
        method_b="fixed",
        min_n_per_group=min_n_per_group,
    )

    subject_level = aggregate_features_by_subject(
        combined,
        feature_cols=feature_cols,
        subject_col="SubjectID" if "SubjectID" in combined.columns else combined.columns[0],
        label_col="Label" if "Label" in combined.columns else "Label",
        method_col="WindowMethod",
    )

    return {
        "combined": combined,
        "feature_columns": pd.DataFrame({"Feature": feature_cols}),
        "comparison": comparison,
        "subject_level": subject_level,
    }
