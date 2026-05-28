# -*- coding: utf-8 -*-
"""
08_nonlinear_visualization_and_rqa_stats.py

我用这个脚本补全论文缺失的 nonlinear dynamics 结果：

1. 3D phase-space attractor examples
   三维相空间吸引子示例：
   (x(t), x(t+tau), x(t+2tau))

2. Publication-ready recurrence plot examples
   论文级 recurrence plot 示例图

3. RQA key metrics statistics
   对核心 RQA 指标做组间统计：
   DET / LAM / L_max / L_mean

4. Event-guided vs fixed key RQA comparison
   对 event-guided 与 fixed-window 的 RQA 表现做汇总图

输入文件依赖：
    results/merged_ml/all_event_subject_features.csv
    results/merged_ml/all_fixed_subject_features.csv

可选原始信号文件：
    results/{SubjectID}/denoised_signal.xlsx
    results/{SubjectID}/main/denoised_signal.xlsx
    results/{SubjectID}/main/{SubjectID}_denoised_signal.xlsx

如果找不到原始信号文件，脚本仍然会继续完成 RQA 统计。
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(r"D:/a_work/课题组实验数据处理/新预处理/results")
MERGED_DIR = ROOT_DIR / "merged_ml"
OUT_DIR = ROOT_DIR / "ml_validation_loso" / "publication_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_FILE = MERGED_DIR / "all_event_subject_features.csv"
FIXED_FILE = MERGED_DIR / "all_fixed_subject_features.csv"


# ============================================================
# Parameters
# ============================================================

HEALTHY_SUBJECT = "27HT000008"
PATIENT_SUBJECT = "51SY000110"

SIGNAL_PREFERENCE = [
    "VMD_Reconstructed",
    "Pressure_Clean",
    "Pressure",
]

N_POINTS = 1200

# RQA / recurrence plot 定量参数：来自参数敏感性分析。
# 主分析使用 4 维嵌入；recurrence density 固定为 5%，方便不同受试者比较。
TAU = 2
EMBED_DIM = 4
RECURRENCE_PERCENTAGE = 5

# 3D attractor 只是可视化投影，所以仍然使用前三个延迟坐标。
ATTRACTOR_DIM = 3

# fixed-window 示例窗口；None 表示取信号中间的一段，避免总是取开头。
FIXED_START = None


# ============================================================
# Basic utilities
# ============================================================

def read_table(path: Path) -> pd.DataFrame:
    """我读取 subject-level 特征表。"""

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def find_signal_file(subject_id: str):
    """我自动寻找每个受试者的去噪信号文件。"""

    subject_dir = ROOT_DIR / subject_id

    candidates = [
        subject_dir / "denoised_signal.xlsx",
        subject_dir / "denoised_signal.csv",
        subject_dir / "main" / "denoised_signal.xlsx",
        subject_dir / "main" / "denoised_signal.csv",
        subject_dir / "main" / f"{subject_id}_denoised_signal.xlsx",
        subject_dir / "main" / f"{subject_id}_denoised_signal.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    # 兜底搜索
    for p in subject_dir.rglob("*denoised*signal*"):
        if p.suffix.lower() in [".xlsx", ".xls", ".csv"]:
            return p

    return None


def load_signal(subject_id: str):
    """我读取某个受试者的压力信号。"""

    path = find_signal_file(subject_id)

    if path is None:
        print(f"Signal file not found for {subject_id}.")
        return None, None

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    signal_col = None

    for key in SIGNAL_PREFERENCE:
        matched = [c for c in df.columns if key in c]
        if matched:
            signal_col = matched[0]
            break

    if signal_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print(f"No numeric signal column found in {path}.")
            return None, None
        signal_col = numeric_cols[-1]

    x = df[signal_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values

    if len(x) < 100:
        print(f"Signal too short for {subject_id}: {len(x)}")
        return None, None

    return x, signal_col


def zscore_signal(x):
    """我对信号做 z-score，方便不同受试者图像比较。"""

    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def make_embedding(x, dim=3, tau=2):
    """我构造延迟嵌入矩阵。"""

    x = np.asarray(x, dtype=float)

    n_vectors = len(x) - (dim - 1) * tau

    if n_vectors <= 0:
        raise ValueError("Signal too short for embedding.")

    emb = np.column_stack([
        x[i * tau: i * tau + n_vectors]
        for i in range(dim)
    ])

    return emb


def recurrence_matrix(x, dim=3, tau=2, percentage=5):
    """
    我生成 recurrence matrix。

    percentage 表示保留距离最小的百分比点，视觉上更适合论文。
    """

    emb = make_embedding(x, dim=dim, tau=tau)

    # pairwise distance
    diff = emb[:, None, :] - emb[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    threshold = np.percentile(dist, percentage)
    R = dist <= threshold

    return R.astype(int)


# ============================================================
# 1. Phase-space attractor plots
# ============================================================

def plot_3d_attractor_examples():
    """我绘制 healthy 与 patient 的三维相空间吸引子。"""

    examples = [
        ("Healthy", HEALTHY_SUBJECT),
        ("Patient", PATIENT_SUBJECT),
    ]

    fig = plt.figure(figsize=(12, 5))

    plotted = 0

    for i, (group_name, subject_id) in enumerate(examples, start=1):

        x, signal_col = load_signal(subject_id)

        if x is None:
            continue

        x = zscore_signal(x[:N_POINTS])
        emb = make_embedding(x, dim=ATTRACTOR_DIM, tau=TAU)

        ax = fig.add_subplot(1, 2, i, projection="3d")

        ax.plot(
            emb[:, 0],
            emb[:, 1],
            emb[:, 2],
            linewidth=0.6,
            alpha=0.85,
        )

        ax.set_title(f"{group_name} attractor\n{subject_id}", fontsize=14)
        ax.set_xlabel(r"$x(t)$")
        ax.set_ylabel(r"$x(t+\tau)$")
        ax.set_zlabel(r"$x(t+2\tau)$")

        ax.grid(True, alpha=0.25)
        plotted += 1

    if plotted > 0:
        plt.tight_layout()
        plt.savefig(
            OUT_DIR / "phase_space_3d_attractor_examples.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
        print("Saved phase_space_3d_attractor_examples.png")
    else:
        plt.close()
        print("No attractor examples saved because signal files were not found.")



# ============================================================
# Event/fixed segment utilities for recurrence plot visualization
# ============================================================

def find_event_feature_file(subject_id: str):
    """寻找某个受试者的 event-guided 窗口特征表。"""

    main_dir = ROOT_DIR / subject_id / "main"

    candidates = []
    if main_dir.exists():
        candidates.extend(main_dir.glob(f"{subject_id}_event_guided_phase_features.*"))

    subject_dir = ROOT_DIR / subject_id
    if subject_dir.exists():
        candidates.extend(subject_dir.rglob(f"{subject_id}_event_guided_phase_features.*"))

    candidates = [p for p in candidates if p.suffix.lower() in [".xlsx", ".xls", ".csv"]]

    return candidates[0] if candidates else None


def read_any_table(path: Path) -> pd.DataFrame:
    """读取 csv/xlsx 表。"""

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def get_fixed_segment(signal: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    """取一个固定长度的 fixed-window 示例片段。"""

    signal = np.asarray(signal, dtype=float)

    if len(signal) <= n_points:
        return signal

    if FIXED_START is None:
        start = max(0, len(signal) // 2 - n_points // 2)
    else:
        start = int(FIXED_START)
        start = min(max(0, start), len(signal) - n_points)

    return signal[start:start + n_points]


def get_event_guided_segment(subject_id: str, signal: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    """
    取 event-guided 示例片段。

    优先从 event-guided feature table 的 StartIndex/EndIndex 中选择长度最接近 n_points 的事件窗口。
    如果事件窗口短于 n_points，则以事件中心向两侧扩展，保证 RP 图尺寸一致。
    如果找不到索引列，则退回到信号中间片段。
    """

    signal = np.asarray(signal, dtype=float)
    event_file = find_event_feature_file(subject_id)

    if event_file is None:
        print(f"Event feature table not found for {subject_id}; use centered segment instead.")
        return get_fixed_segment(signal, n_points=n_points)

    event_df = read_any_table(event_file)

    if "StartIndex" not in event_df.columns or "EndIndex" not in event_df.columns:
        print(f"No StartIndex/EndIndex in {event_file.name}; use centered segment instead.")
        return get_fixed_segment(signal, n_points=n_points)

    tmp = event_df[["StartIndex", "EndIndex"]].copy()
    tmp["StartIndex"] = pd.to_numeric(tmp["StartIndex"], errors="coerce")
    tmp["EndIndex"] = pd.to_numeric(tmp["EndIndex"], errors="coerce")
    tmp = tmp.dropna()
    tmp["Length"] = tmp["EndIndex"] - tmp["StartIndex"]
    tmp = tmp[tmp["Length"] > 10]

    if tmp.empty:
        print(f"No valid event index rows for {subject_id}; use centered segment instead.")
        return get_fixed_segment(signal, n_points=n_points)

    tmp["DiffToN"] = (tmp["Length"] - n_points).abs()
    row = tmp.sort_values("DiffToN").iloc[0]

    start = int(row["StartIndex"])
    end = int(row["EndIndex"])
    center = (start + end) // 2

    if len(signal) <= n_points:
        return signal

    start2 = max(0, center - n_points // 2)
    start2 = min(start2, len(signal) - n_points)
    end2 = start2 + n_points

    return signal[start2:end2]

# ============================================================
# 2. Recurrence plot examples
# ============================================================

def plot_recurrence_examples():
    """
    绘制 2×2 recurrence plot：
        上排：event-guided Healthy vs STC
        下排：fixed-window Healthy vs STC

    这张图用于同时展示：
    1. Healthy 与 STC 的 recurrence topology 差异；
    2. event-guided 与 fixed-window 对动力学结构呈现的影响。
    """

    subjects = {
        "Healthy": HEALTHY_SUBJECT,
        "STC": PATIENT_SUBJECT,
    }

    segments = {}

    for group_name, subject_id in subjects.items():
        x, signal_col = load_signal(subject_id)

        if x is None:
            print(f"Skip {group_name}: signal not found.")
            continue

        fixed_seg = get_fixed_segment(x, n_points=N_POINTS)
        event_seg = get_event_guided_segment(subject_id, x, n_points=N_POINTS)

        segments[("Event-guided", group_name)] = event_seg
        segments[("Fixed-window", group_name)] = fixed_seg

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    plot_layout = [
        (0, 0, "Event-guided", "Healthy"),
        (0, 1, "Event-guided", "STC"),
        (1, 0, "Fixed-window", "Healthy"),
        (1, 1, "Fixed-window", "STC"),
    ]

    plotted = 0

    for row, col, method_name, group_name in plot_layout:
        ax = axes[row, col]
        key = (method_name, group_name)
        subject_id = subjects[group_name]

        if key not in segments:
            ax.axis("off")
            ax.set_title(f"{method_name} {group_name}\nsignal not found")
            continue

        seg = zscore_signal(segments[key])

        if len(seg) <= (EMBED_DIM - 1) * TAU:
            ax.axis("off")
            ax.set_title(f"{method_name} {group_name}\nsegment too short")
            continue

        R = recurrence_matrix(
            seg,
            dim=EMBED_DIM,
            tau=TAU,
            percentage=RECURRENCE_PERCENTAGE,
        )

        ax.imshow(
            R,
            cmap="binary",
            origin="lower",
            interpolation="nearest",
        )

        ax.set_title(f"{method_name} {group_name}\n{subject_id}", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        plotted += 1

    fig.suptitle(
        f"Representative recurrence plots (m={EMBED_DIM}, τ={TAU}, RR={RECURRENCE_PERCENTAGE}%)",
        fontsize=16,
        y=0.98,
    )

    if plotted > 0:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_2x2 = OUT_DIR / "recurrence_plot_event_vs_fixed_2x2.png"
        plt.savefig(out_2x2, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_2x2.name}")
    else:
        plt.close()
        print("No recurrence examples saved because signal files were not found.")



# ============================================================
# 3. RQA statistics
# ============================================================

def get_rqa_columns(df: pd.DataFrame):
    """我提取核心 RQA 指标列。"""

    key_patterns = [
        "RQA_DET",
        "RQA_LAM",
        "RQA_L_max",
        "RQA_L_mean",
        "RQA_RR",
        "RQA_TT",
    ]

    cols = [
        c for c in df.columns
        if any(p in c for p in key_patterns)
        and "epsilon" not in c
    ]

    return cols


def infer_metric_group(feature_name: str):
    """我从列名中推断 RQA 指标类别。"""

    for metric in ["RQA_DET", "RQA_LAM", "RQA_L_max", "RQA_L_mean", "RQA_RR", "RQA_TT"]:
        if metric in feature_name:
            return metric

    return "Other"


def mannwhitney_feature_stats(df: pd.DataFrame, method_name: str, cols):
    """我对每个 RQA 特征做 healthy vs patient 的 Mann-Whitney U 检验。"""

    rows = []

    for col in cols:

        sub = df[["SubjectID", "Label", col]].dropna()

        if sub["Label"].nunique() < 2:
            continue

        healthy = sub[sub["Label"] == 0][col].values
        patient = sub[sub["Label"] == 1][col].values

        if len(healthy) < 2 or len(patient) < 2:
            continue

        try:
            stat, p = mannwhitneyu(
                healthy,
                patient,
                alternative="two-sided",
            )
        except Exception:
            stat, p = np.nan, np.nan

        # rank-biserial correlation effect size
        n0 = len(healthy)
        n1 = len(patient)
        rbc = 1 - (2 * stat) / (n0 * n1) if np.isfinite(stat) else np.nan

        rows.append({
            "Method": method_name,
            "Feature": col,
            "MetricGroup": infer_metric_group(col),
            "Healthy_median": np.median(healthy),
            "Patient_median": np.median(patient),
            "Healthy_mean": np.mean(healthy),
            "Patient_mean": np.mean(patient),
            "U": stat,
            "p_value": p,
            "Effect_rank_biserial": rbc,
            "N_healthy": n0,
            "N_patient": n1,
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out["abs_effect"] = out["Effect_rank_biserial"].abs()
        out = out.sort_values(["p_value", "abs_effect"], ascending=[True, False])

    return out


def aggregate_metric_group_stats(stats_df: pd.DataFrame):
    """
    我把单个 RQA 特征统计汇总到 RQA 指标组层面。
    例如 RQA_DET 下可能有 mean/std/median/iqr 和多个信号源。
    """

    if stats_df.empty:
        return pd.DataFrame()

    rows = []

    for (method, metric), sub in stats_df.groupby(["Method", "MetricGroup"]):

        rows.append({
            "Method": method,
            "MetricGroup": metric,
            "N_features": len(sub),
            "Median_abs_effect": sub["abs_effect"].median(),
            "Max_abs_effect": sub["abs_effect"].max(),
            "Min_p_value": sub["p_value"].min(),
            "N_p_less_0.05": int((sub["p_value"] < 0.05).sum()),
        })

    return pd.DataFrame(rows).sort_values(
        ["Method", "Median_abs_effect"],
        ascending=[True, False],
    )


def plot_top_rqa_effects(stats_df: pd.DataFrame):
    """我绘制 top RQA 特征效应量图。"""

    if stats_df.empty:
        print("No RQA stats to plot.")
        return

    top = stats_df.sort_values("abs_effect", ascending=False).head(20).iloc[::-1]

    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    plt.barh(top["Feature"], top["Effect_rank_biserial"])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Rank-biserial effect size")
    plt.title("Top RQA group differences")
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "top_rqa_group_difference_effects.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved top_rqa_group_difference_effects.png")


def plot_metric_group_summary(group_df: pd.DataFrame):
    """我绘制 RQA 指标组的中位效应量汇总图。"""

    if group_df.empty:
        return

    pivot = group_df.pivot(
        index="MetricGroup",
        columns="Method",
        values="Median_abs_effect",
    ).fillna(0)

    pivot = pivot.sort_values(
        by=pivot.columns.tolist(),
        ascending=False,
    )

    ax = pivot.plot(
        kind="barh",
        figsize=(8, 5),
        width=0.75,
    )

    ax.set_xlabel("Median absolute rank-biserial effect size")
    ax.set_ylabel("RQA metric group")
    ax.set_title("RQA metric group effect summary")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "rqa_metric_group_effect_summary.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved rqa_metric_group_effect_summary.png")


def run_rqa_statistics():
    """我运行 event-guided 与 fixed-window 的 RQA 统计。"""

    event_df = read_table(EVENT_FILE)
    fixed_df = read_table(FIXED_FILE)

    all_stats = []

    for method, df in [
        ("event_guided", event_df),
        ("fixed", fixed_df),
    ]:

        rqa_cols = get_rqa_columns(df)
        print(f"{method}: found {len(rqa_cols)} RQA columns.")

        stats_df = mannwhitney_feature_stats(
            df=df,
            method_name=method,
            cols=rqa_cols,
        )

        all_stats.append(stats_df)

    all_stats_df = pd.concat(all_stats, ignore_index=True)

    all_stats_df.to_csv(
        OUT_DIR / "rqa_feature_group_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_summary = aggregate_metric_group_stats(all_stats_df)

    group_summary.to_csv(
        OUT_DIR / "rqa_metric_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_top_rqa_effects(all_stats_df)
    plot_metric_group_summary(group_summary)

    print("Saved RQA statistics tables.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("1. 3D phase-space attractor examples")
    print("=" * 70)
    plot_3d_attractor_examples()

    print("\n" + "=" * 70)
    print("2. Recurrence plot examples")
    print("=" * 70)
    plot_recurrence_examples()

    print("\n" + "=" * 70)
    print("3. RQA group statistics")
    print("=" * 70)
    run_rqa_statistics()

    print("\nFinished.")
    print("Saved to:", OUT_DIR.resolve())
