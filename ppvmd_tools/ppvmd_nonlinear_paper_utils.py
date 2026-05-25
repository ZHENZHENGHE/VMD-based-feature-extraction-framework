# -*- coding: utf-8 -*-
"""
ppvmd_nonlinear_paper_utils.py

我把论文阶段常用的非线性动力学辅助函数集中放在这里。
这个文件不改动既有 ppvmd_tools 里的预处理、VMD、相空间特征提取函数；
它只负责论文图、RQA 可视化、分形复杂度、统计比较和受试者级输出。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, ttest_ind


# ============================================================
# 1. Signal selection
# ============================================================


def get_available_signal_columns(
    df: pd.DataFrame,
    preferred: Sequence[str] = (
        "Pressure_Clean",
        "VMD_Reconstructed",
        "VMD_Mode3",
        "VMD_Mode4",
    ),
) -> List[str]:
    """我只返回当前数据表中真实存在的信号列，避免后面画图时报错。"""
    return [c for c in preferred if c in df.columns]


# ============================================================
# 2. Phase-space visualization
# ============================================================


def plot_phase_space_grid(
    phase_df: pd.DataFrame,
    windows: Sequence,
    signal_col: str,
    estimate_embedding_parameters,
    phase_space_reconstruct,
    num_show: int = 6,
    max_tau: int = 120,
    max_dim: int = 8,
    ncols: int = 3,
    title_prefix: str = "",
):
    """
    我把多个窗口的三维相空间轨迹画成网格图。

    我在这里每个窗口单独估计 tau 和 m，用于探索性可视化。
    最终分类阶段可以在训练集中固定 tau/m，以避免信息泄露。
    """
    if signal_col not in phase_df.columns:
        print(f"Skip {signal_col}: column not found.")
        return None

    num_show = min(num_show, len(windows))
    if num_show == 0:
        print(f"No windows available for {signal_col}.")
        return None

    nrows = math.ceil(num_show / ncols)
    fig = plt.figure(figsize=(5.2 * ncols, 4.6 * nrows))

    for i in range(num_show):
        w = windows[i]
        x = np.asarray(phase_df[signal_col].values[w.start_idx:w.end_idx], dtype=float)
        x = x[np.isfinite(x)]

        if len(x) < 30:
            continue

        tau, m, _ = estimate_embedding_parameters(
            x,
            tau_method="ami",
            max_tau=max_tau,
            max_dim=max_dim,
        )
        tau = int(max(1, tau))
        m = int(max(3, m))
        m_plot = max(3, m)

        x_norm = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
        X = phase_space_reconstruct(x_norm, m=m_plot, tau=tau)

        if X is None or X.shape[1] < 3 or len(X) <= 10:
            continue

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot(X[:, 0], X[:, 1], X[:, 2], linewidth=0.7)
        ax.set_title(f"Window {w.window_id}\ntau={tau}, m={m}", fontsize=10)
        ax.set_xlabel("x(t)", fontsize=8)
        ax.set_ylabel(f"x(t+{tau})", fontsize=8)
        ax.set_zlabel("x(t+2τ)", fontsize=8)

    fig.suptitle(f"{title_prefix} Phase-space trajectories: {signal_col}", fontsize=16)
    plt.tight_layout()
    return fig


# ============================================================
# 3. Recurrence plot visualization
# ============================================================


def reconstruct_phase_space_for_rp(x: Sequence[float], tau: int, m: int) -> Optional[np.ndarray]:
    """我按延迟嵌入公式构造 recurrence plot 使用的相空间矩阵。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    tau = int(max(1, tau))
    m = int(max(1, m))
    n_vectors = n - (m - 1) * tau

    if n_vectors <= 10:
        return None

    return np.column_stack([x[i * tau : i * tau + n_vectors] for i in range(m)])


def recurrence_matrix_fixed_rr(X: np.ndarray, rr: float = 0.05) -> Tuple[Optional[np.ndarray], float]:
    """
    我用固定 recurrence rate 反推出 epsilon，再构造 recurrence matrix。

    这样做的原因是：不同窗口的 recurrence density 一致，图之间更公平可比。
    """
    X = np.asarray(X, dtype=float)
    if len(X) < 10:
        return None, np.nan

    dist = squareform(pdist(X, metric="euclidean"))
    d = dist[np.triu_indices_from(dist, k=1)]
    d = d[np.isfinite(d)]

    if len(d) == 0:
        return None, np.nan

    epsilon = float(np.quantile(d, rr))
    R = dist <= epsilon
    np.fill_diagonal(R, False)
    return R.astype(int), epsilon


def estimate_tau_m_safe(
    x: Sequence[float],
    estimate_embedding_parameters,
    max_tau: int = 120,
    max_dim: int = 8,
) -> Tuple[int, int]:
    """我给 tau/m 估计加保护，防止个别窗口失败导致整批绘图中断。"""
    try:
        tau, m, _ = estimate_embedding_parameters(
            x,
            tau_method="ami",
            max_tau=max_tau,
            max_dim=max_dim,
        )
        return int(max(1, tau)), int(max(3, m))
    except Exception as exc:
        print("Embedding parameter estimation failed:", exc)
        return 5, 3


def plot_recurrence_grid(
    phase_df: pd.DataFrame,
    windows: Sequence,
    signal_col: str,
    estimate_embedding_parameters,
    num_show: int = 6,
    rr: float = 0.05,
    max_tau: int = 120,
    max_dim: int = 8,
    ncols: int = 3,
    title_prefix: str = "Event-guided",
):
    """我用固定 recurrence rate 画多个窗口的 recurrence plots。"""
    if signal_col not in phase_df.columns:
        print(f"Skip {signal_col}: column not found.")
        return None

    num_show = min(num_show, len(windows))
    if num_show == 0:
        print(f"No windows available for {signal_col}.")
        return None

    nrows = math.ceil(num_show / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for i in range(num_show):
        ax = axes[i]
        w = windows[i]
        x = np.asarray(phase_df[signal_col].values[w.start_idx:w.end_idx], dtype=float)
        x = x[np.isfinite(x)]

        if len(x) < 30:
            ax.axis("off")
            ax.set_title(f"Window {w.window_id}\nToo short")
            continue

        tau, m = estimate_tau_m_safe(
            x,
            estimate_embedding_parameters=estimate_embedding_parameters,
            max_tau=max_tau,
            max_dim=max_dim,
        )

        x_norm = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
        X = reconstruct_phase_space_for_rp(x_norm, tau=tau, m=m)
        if X is None:
            ax.axis("off")
            ax.set_title(f"Window {w.window_id}\nEmbedding failed")
            continue

        R, epsilon = recurrence_matrix_fixed_rr(X, rr=rr)
        if R is None:
            ax.axis("off")
            ax.set_title(f"Window {w.window_id}\nRP failed")
            continue

        ax.imshow(R, cmap="binary", origin="lower", interpolation="nearest")
        ax.set_title(f"Window {w.window_id}\ntau={tau}, m={m}, eps={epsilon:.3f}", fontsize=9)
        ax.set_xlabel("i", fontsize=8)
        ax.set_ylabel("j", fontsize=8)

    for j in range(num_show, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title_prefix} Recurrence Plots: {signal_col}", fontsize=16)
    plt.tight_layout()
    return fig


# ============================================================
# 4. Fractal complexity features
# ============================================================


def higuchi_fd(x: Sequence[float], kmax: int = 10) -> float:
    """我计算 Higuchi fractal dimension，用它描述短窗口信号的跨尺度粗糙度。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 50:
        return np.nan

    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            diff = np.abs(np.diff(x[idx])).sum()
            norm = (N - 1) / (len(idx) * k)
            Lk.append(diff * norm / k)
        L.append(np.nanmean(Lk) if len(Lk) else np.nan)

    L = np.asarray(L, dtype=float)
    k_values = np.arange(1, kmax + 1)
    valid = np.isfinite(L) & (L > 0)
    if valid.sum() < 3:
        return np.nan

    coeffs = np.polyfit(np.log(1.0 / k_values[valid]), np.log(L[valid]), 1)
    return float(coeffs[0])


def katz_fd(x: Sequence[float]) -> float:
    """我计算 Katz fractal dimension，作为 HFD 的辅助分形指标。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 3:
        return np.nan

    L = np.sum(np.abs(np.diff(x)))
    d = np.max(np.abs(x - x[0]))
    if L <= 1e-12 or d <= 1e-12:
        return np.nan
    return float(np.log10(N) / (np.log10(d / L) + np.log10(N)))


def petrosian_fd(x: Sequence[float]) -> float:
    """我计算 Petrosian fractal dimension，作为快速稳健的辅助复杂度指标。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 3:
        return np.nan

    dx = np.diff(x)
    sign_changes = np.sum(dx[1:] * dx[:-1] < 0)
    if sign_changes == 0:
        return 1.0
    return float(np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * sign_changes))))


def robust_zscore_1d(x: Sequence[float]) -> np.ndarray:
    """我用 median 和 MAD 做稳健标准化，避免异常点主导分形特征。"""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return (x - med) / (1.4826 * mad + 1e-12)


def add_fractal_features_to_feature_table(
    feature_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    windows: Sequence,
    signal_cols: Sequence[str],
    kmax: int = 10,
) -> pd.DataFrame:
    """我给已有窗口特征表追加 HFD/KFD/PFD 三类分形复杂度特征。"""
    out = feature_df.copy()

    for i, w in enumerate(windows):
        if i >= len(out):
            break
        for col in signal_cols:
            if col not in phase_df.columns:
                continue
            x = np.asarray(phase_df[col].values[w.start_idx:w.end_idx], dtype=float)
            x = x[np.isfinite(x)]

            if len(x) < 50:
                hfd = kfd = pfd = np.nan
            else:
                x_norm = robust_zscore_1d(x)
                hfd = higuchi_fd(x_norm, kmax=kmax)
                kfd = katz_fd(x_norm)
                pfd = petrosian_fd(x_norm)

            out.loc[out.index[i], f"{col}_HFD"] = hfd
            out.loc[out.index[i], f"{col}_KFD"] = kfd
            out.loc[out.index[i], f"{col}_PFD"] = pfd

    return out


# ============================================================
# 5. Statistical comparison
# ============================================================


def get_default_nonlinear_feature_suffixes() -> Tuple[str, ...]:
    """我统一定义论文阶段关注的非线性特征后缀。"""
    return (
        "SampEn", "PermEn",
        "PS_TrajectoryLength", "PS_MeanStep", "PS_StepStd", "PS_StateSpread",
        "PS_Eig1", "PS_Eig2", "PS_EigRatio12", "PS_LogVolume",
        "RQA_RR", "RQA_DET", "RQA_LAM", "RQA_L_mean", "RQA_L_max",
        "RQA_V_mean", "RQA_TT", "RQA_epsilon",
        "ApproxLLE", "HFD", "KFD", "PFD",
    )


def infer_nonlinear_feature_columns(
    df: pd.DataFrame,
    signal_prefixes: Optional[Sequence[str]] = None,
    suffixes: Optional[Sequence[str]] = None,
) -> List[str]:
    """我从特征表里自动识别可用于统计比较的数值特征列。"""
    if suffixes is None:
        suffixes = get_default_nonlinear_feature_suffixes()

    cols = []
    for col in df.columns:
        if signal_prefixes is not None and not any(col.startswith(prefix + "_") for prefix in signal_prefixes):
            continue
        if any(col.endswith("_" + s) or col.endswith(s) for s in suffixes):
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().sum() > 0:
                cols.append(col)
    return cols


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """我计算 Cohen's d，用标准化均值差表示两组差异强度。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / max(len(x) + len(y) - 2, 1)
    if pooled <= 1e-12:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """我计算 Cliff's delta，用非参数秩效应量表示两组整体大小关系。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(x) * len(y)))


def benjamini_hochberg_fdr(p_values: Sequence[float]) -> np.ndarray:
    """我用 Benjamini-Hochberg 方法做 FDR 多重比较校正。"""
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


def balanced_sample_fixed_windows(
    event_feature_df: pd.DataFrame,
    fixed_feature_df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """我从 fixed 窗口中随机抽取与 event-guided 数量相同的窗口，形成平衡比较表。"""
    event_df = event_feature_df.copy()
    fixed_df = fixed_feature_df.copy()
    event_df["WindowMethod"] = "event_guided"
    fixed_df["WindowMethod"] = "fixed"

    if len(fixed_df) > len(event_df):
        fixed_sampled = fixed_df.sample(n=len(event_df), replace=False, random_state=random_state)
    else:
        fixed_sampled = fixed_df.copy()

    combined = pd.concat([event_df, fixed_sampled], ignore_index=True)
    combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return event_df, fixed_sampled, combined


def compare_two_window_methods(
    feature_df: pd.DataFrame,
    feature_cols: Sequence[str],
    method_col: str = "WindowMethod",
    method_a: str = "event_guided",
    method_b: str = "fixed",
    min_n_per_group: int = 3,
) -> pd.DataFrame:
    """我比较 event-guided 与 fixed 两种窗口方法的非线性特征差异。"""
    df = feature_df.copy()
    rows = []

    for feature in feature_cols:
        if feature not in df.columns:
            continue
        xa = pd.to_numeric(df.loc[df[method_col] == method_a, feature], errors="coerce").dropna().to_numpy(float)
        xb = pd.to_numeric(df.loc[df[method_col] == method_b, feature], errors="coerce").dropna().to_numpy(float)

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
        result["FDR_q"] = benjamini_hochberg_fdr(result["MannWhitney_p"].to_numpy(float))
        result["AbsCliffsDelta"] = result["CliffsDelta"].abs()
        result["AbsCohenD"] = result["CohenD"].abs()
        result = result.sort_values(["AbsCliffsDelta", "AbsCohenD"], ascending=False).reset_index(drop=True)
    return result


def build_event_vs_fixed_report(
    event_feature_df: pd.DataFrame,
    fixed_feature_df: pd.DataFrame,
    signal_prefixes: Sequence[str] = ("Pressure_Clean", "VMD_Reconstructed", "VMD_Mode3", "VMD_Mode4"),
    random_state: int = 42,
    min_n_per_group: int = 3,
) -> Dict[str, pd.DataFrame]:
    """我一键生成 event-guided vs fixed 的平衡统计报告。"""
    event_df, fixed_sampled_df, combined = balanced_sample_fixed_windows(
        event_feature_df,
        fixed_feature_df,
        random_state=random_state,
    )
    feature_cols = infer_nonlinear_feature_columns(combined, signal_prefixes=signal_prefixes)
    comparison = compare_two_window_methods(
        combined,
        feature_cols=feature_cols,
        method_col="WindowMethod",
        method_a="event_guided",
        method_b="fixed",
        min_n_per_group=min_n_per_group,
    )
    return {
        "event": event_df,
        "fixed_sampled": fixed_sampled_df,
        "combined": combined,
        "feature_columns": pd.DataFrame({"Feature": feature_cols}),
        "comparison": comparison,
    }


# ============================================================
# 6. Plotting statistics
# ============================================================


def plot_method_boxplots(
    feature_df: pd.DataFrame,
    features: Sequence[str],
    method_col: str = "WindowMethod",
    method_order: Sequence[str] = ("event_guided", "fixed"),
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5.0, 4.0),
    show_points: bool = True,
    title: str = "Event-guided vs fixed-window nonlinear dynamics features",
):
    """我为核心特征画 event-guided 与 fixed 的箱线图。"""
    features = [f for f in features if f in feature_df.columns]
    if len(features) == 0:
        raise ValueError("No features available for plotting.")

    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes = axes.reshape(-1)
    rng = np.random.default_rng(42)

    for i, feature in enumerate(features):
        ax = axes[i]
        data = []
        labels = []
        for method in method_order:
            values = pd.to_numeric(feature_df.loc[feature_df[method_col] == method, feature], errors="coerce")
            values = values.dropna().to_numpy(float)
            data.append(values)
            labels.append(method)
        ax.boxplot(data, labels=labels, showfliers=True)
        if show_points:
            for j, values in enumerate(data, start=1):
                if len(values):
                    ax.scatter(np.full(len(values), j) + rng.normal(0, 0.035, len(values)), values, s=18, alpha=0.7)
        ax.set_title(feature, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)

    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    return fig


def plot_top_method_differences(
    comparison_df: pd.DataFrame,
    top_n: int = 20,
    effect_col: str = "CliffsDelta",
    title: str = "Top event-guided vs fixed-window differences by Cliff's delta",
):
    """我按效应量排序画条形图，帮助筛选最敏感的非线性指标。"""
    if comparison_df.empty:
        raise ValueError("comparison_df is empty.")
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
# 7. Subject-level table
# ============================================================


def aggregate_features_by_subject(
    feature_df: pd.DataFrame,
    feature_cols: Sequence[str],
    subject_col: str = "SubjectID",
    label_col: str = "Label",
    method_col: str = "WindowMethod",
    aggregations: Sequence[str] = ("mean", "std", "median", "min", "max"),
) -> pd.DataFrame:
    """我把窗口级特征聚合为受试者级特征，避免把同一个人的多个窗口当成独立受试者。"""
    df = feature_df.copy()
    group_cols = [subject_col]
    if label_col in df.columns:
        group_cols.append(label_col)
    if method_col in df.columns:
        group_cols.append(method_col)

    available_features = [c for c in feature_cols if c in df.columns]
    grouped = df.groupby(group_cols, dropna=False)[available_features].agg(list(aggregations))
    grouped.columns = [f"{feat}__{agg}" for feat, agg in grouped.columns]
    out = grouped.reset_index()
    counts = df.groupby(group_cols, dropna=False).size().reset_index(name="NumWindows")
    return out.merge(counts, on=group_cols, how="left")


def build_subject_master_features(
    subject_id: str,
    label: int,
    event_feature_df: pd.DataFrame,
    fixed_feature_df: pd.DataFrame,
    output_root: str | Path = "./results",
    fixed_method_name: str = "fixed300",
) -> pd.DataFrame:
    """我把一个受试者的 event-guided 和 fixed-window 特征合并成 master_features 表。"""
    event_df = event_feature_df.copy()
    fixed_df = fixed_feature_df.copy()

    event_df["SubjectID"] = subject_id
    event_df["Label"] = label
    event_df["WindowMethod"] = "event_guided"
    fixed_df["SubjectID"] = subject_id
    fixed_df["Label"] = label
    fixed_df["WindowMethod"] = fixed_method_name

    master_df = pd.concat([event_df, fixed_df], ignore_index=True)
    front_cols = [
        "SubjectID", "Label", "WindowMethod", "WindowID", "StartTime", "EndTime", "CenterTime",
        "StartIndex", "EndIndex", "NumPoints", "EventID", "EventStartTime", "EventEndTime", "EventCoverage",
    ]
    existing_front_cols = [c for c in front_cols if c in master_df.columns]
    other_cols = [c for c in master_df.columns if c not in existing_front_cols]
    master_df = master_df[existing_front_cols + other_cols]

    subject_dir = Path(output_root) / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    output_path = subject_dir / f"{subject_id}_master_features.csv"
    master_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Saved master feature table:")
    print(output_path)
    print("Shape:", master_df.shape)
    print(master_df["WindowMethod"].value_counts())
    return master_df
