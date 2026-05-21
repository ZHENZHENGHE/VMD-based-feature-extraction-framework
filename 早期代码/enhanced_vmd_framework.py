# -*- coding: utf-8 -*-
"""
enhanced_vmd_framework.py

目的：
    在你现有的 physiology_preserving_adaptive_denoise 框架基础上，补齐审稿人最容易质疑的部分：
    1) 半合成 ground-truth 数据集，用于客观评价去噪和伪影识别；
    2) 新增评价指标：伪影检测、事件起止时间误差、clean-signal RMSE 等；
    3) 消融实验框架，证明每个模块都有贡献；
    4) 信号质量驱动的自适应参数推荐；
    5) 事件保护型 VMD 模态选择，而不是简单“VMD分解后人工看图选模态”。

依赖：
    numpy, pandas, scipy
    可选：vmdpy。如果没有安装 vmdpy，VMD相关函数会给出明确报错。

建议用法：
    from enhanced_vmd_framework import *

    # 1. 先用你的原始算法得到 df_out
    # from vmd_utils import physiology_preserving_adaptive_denoise
    # df_out = physiology_preserving_adaptive_denoise(data, ...)

    # 2. 构造半合成数据
    # semi_df, truth = make_semisynthetic_pressure_dataset(df_out)

    # 3. 在半合成数据上运行你的算法
    # pred_df = physiology_preserving_adaptive_denoise(semi_df, pressure_col='Pressure', ...)

    # 4. 计算客观指标
    # report = evaluate_semisynthetic_result(pred_df, truth)

    # 5. VMD事件保护型模态选择
    # vmd_res = event_preserving_vmd_denoise(
    #     signal=df_out['Pressure_Preclean'].values,
    #     time=df_out['Time'].values,
    #     event_mask=df_out['PhysioEventMask'].values,
    #     artifact_mask=df_out['ArtifactMask'].values,
    #     K=5,
    #     alpha=2700,
    # )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any

from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq


# ============================================================
# 0. 基础工具函数
# ============================================================

def robust_mad(x: np.ndarray, scale: bool = True, eps: float = 1e-12) -> float:
    """
    计算 robust MAD（Median Absolute Deviation）。

    为什么用 MAD：
        压力信号中可能存在尖刺、掉点、离群值。
        普通标准差会被极端值拉大，导致后续阈值失真。
        MAD 对离群值更稳健。

    参数：
        x     : 输入数组
        scale : True 时乘以 1.4826，使 MAD 在高斯噪声下近似标准差
        eps   : 防止返回 0 后导致除零

    返回：
        float, 稳健噪声尺度
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan

    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if scale:
        mad *= 1.4826
    return float(max(mad, eps))


def estimate_dt(time: np.ndarray) -> float:
    """
    根据时间列估计采样间隔，使用中位数而不是平均数，避免异常间隔影响。
    """
    t = np.asarray(time, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Time 至少需要两个递增点。")
    return float(np.median(dt))


def components(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    返回布尔 mask 中连续 True 的区间。

    返回格式：
        [(start, end), ...]
    其中 end 是开区间，不包含 end。
    """
    mask = np.asarray(mask, dtype=bool)
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        s = i
        while i < n and mask[i]:
            i += 1
        out.append((s, i))
    return out


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """
    中位数滑窗，用于估计慢变化 baseline。
    """
    window = int(max(window, 3))
    if window % 2 == 0:
        window += 1
    return (
        pd.Series(np.asarray(x, dtype=float))
        .rolling(window=window, center=True, min_periods=max(3, window // 3))
        .median()
        .bfill()
        .ffill()
        .to_numpy()
    )


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    安全相关系数计算。
    如果有效点太少或方差接近 0，返回 np.nan。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    计算两个 mask 的 Intersection-over-Union。
    常用于事件区间或伪影区间识别的重叠评价。
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return np.nan
    return float(inter / union)


# ============================================================
# 1. 信号质量驱动的自适应参数推荐
# ============================================================

def recommend_adaptive_parameters(
    data: pd.DataFrame,
    time_col: str = "Time",
    pressure_col: str = "Pressure",
    pressure_min: float = 85.0,
    pressure_max: float = 130.0,
) -> Dict[str, float]:
    """
    根据信号自身质量自动推荐核心参数。

    解决审稿人质疑：
        “你的 event_amp_k、amp_k、smooth_window_s 是否只是人工经验？”

    思路：
        1. 用一阶差分 MAD 衡量高频噪声；
        2. 用原始压力 MAD 衡量整体波动；
        3. 用超出生理范围的比例衡量严重异常程度；
        4. 根据信号质量分级动态调整阈值和窗口。

    返回：
        可以直接传给 physiology_preserving_adaptive_denoise 的参数字典。
    """
    df = data[[time_col, pressure_col]].replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values(time_col)

    t = df[time_col].to_numpy(dtype=float)
    p = df[pressure_col].to_numpy(dtype=float)
    dt = estimate_dt(t)

    dp = np.diff(p)
    pressure_scale = robust_mad(p)
    diff_scale = robust_mad(dp)

    # 超出生理范围的点比例，可作为“异常污染程度”的粗略估计。
    range_abnormal_ratio = np.mean((p < pressure_min) | (p > pressure_max))

    # 高频噪声相对整体压力波动的比例。
    # 比例越高，说明信号越不平滑，伪影/噪声越重。
    noise_ratio = diff_scale / (pressure_scale + 1e-9)

    # 将信号质量映射到 0~1。
    # 这里不是分类器，而是一个透明、可解释的质量评分。
    quality_badness = np.clip(0.65 * noise_ratio + 3.0 * range_abnormal_ratio, 0.0, 1.0)

    # 噪声越重，伪影检测阈值可略降低，平滑窗口略加大。
    # 但事件检测阈值不能过低，否则会把噪声误判成生理事件。
    amp_k = 6.5 - 1.5 * quality_badness
    slope_k = 6.5 - 1.5 * quality_badness
    curvature_k = 6.5 - 1.5 * quality_badness

    event_amp_k = 3.0 + 0.8 * quality_badness
    event_area_k = 1.5 + 0.5 * quality_badness

    # 窗口以秒为单位，自动与采样率解耦。
    smooth_window_s = max(3.0 * dt, 4.8 + 3.0 * quality_badness)
    noise_window_s = max(10.0 * dt, 20.0 + 10.0 * quality_badness)

    return {
        "amp_k": float(amp_k),
        "slope_k": float(slope_k),
        "curvature_k": float(curvature_k),
        "event_amp_k": float(event_amp_k),
        "event_area_k": float(event_area_k),
        "smooth_window_s": float(smooth_window_s),
        "noise_window_s": float(noise_window_s),
        "signal_quality_badness": float(quality_badness),
        "range_abnormal_ratio": float(range_abnormal_ratio),
        "noise_ratio": float(noise_ratio),
    }


# ============================================================
# 2. 半合成 ground-truth 数据集
# ============================================================

@dataclass
class SemiSyntheticTruth:
    """
    半合成数据的真值容器。

    clean_pressure:
        注入伪影和噪声之前的“干净参考信号”。

    observed_pressure:
        加入噪声和伪影后的观测压力，算法应该处理这个信号。

    artifact_mask:
        人工注入伪影的位置真值。

    event_mask:
        生理事件位置真值。

    artifact_table:
        每个注入伪影的类型、起止位置、幅度等信息。
    """
    clean_pressure: np.ndarray
    observed_pressure: np.ndarray
    artifact_mask: np.ndarray
    event_mask: np.ndarray
    artifact_table: pd.DataFrame


def _infer_clean_reference(
    df: pd.DataFrame,
    time_col: str,
    pressure_col: str,
    event_col: Optional[str],
    baseline_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从已有数据推断一个“相对干净”的参考信号。

    如果 df 已经来自你的预处理结果，优先用 Pressure_Preclean 或 Pressure_Clean。
    如果只有原始 Pressure，则用滚动中位数 + 轻微 SG 平滑生成参考。
    """
    if "Pressure_Preclean" in df.columns:
        clean = df["Pressure_Preclean"].to_numpy(dtype=float)
    elif "Pressure_Clean" in df.columns:
        clean = df["Pressure_Clean"].to_numpy(dtype=float)
    else:
        p = df[pressure_col].to_numpy(dtype=float)
        t = df[time_col].to_numpy(dtype=float)
        dt = estimate_dt(t)
        w = int(round(9.0 / dt))
        w = max(w, 7)
        if w % 2 == 0:
            w += 1
        w = min(w, len(p) - 1 if (len(p) - 1) % 2 == 1 else len(p) - 2)
        if w >= 7:
            clean = savgol_filter(p, window_length=w, polyorder=2, mode="interp")
        else:
            clean = p.copy()

    if event_col is not None and event_col in df.columns:
        event_mask = df[event_col].to_numpy(dtype=bool)
    elif "PhysioEventMask" in df.columns:
        event_mask = df["PhysioEventMask"].to_numpy(dtype=bool)
    else:
        # 如果没有事件真值，则用 baseline 残差自适应估计。
        t = df[time_col].to_numpy(dtype=float)
        dt = estimate_dt(t)
        w = int(round(60.0 / dt))
        w = max(w, 9)
        if w % 2 == 0:
            w += 1
        baseline = rolling_median(clean, w)
        residual = clean - baseline
        sigma = robust_mad(np.diff(clean)) / np.sqrt(2.0)
        event_mask = residual > 3.0 * sigma

    return clean, event_mask


def make_semisynthetic_pressure_dataset(
    df: pd.DataFrame,
    time_col: str = "Time",
    pressure_col: str = "Pressure",
    ph_col: str = "PH",
    temp_col: str = "Temperature",
    event_col: Optional[str] = "PhysioEventMask",
    baseline_col: Optional[str] = "Baseline",
    random_state: int = 42,
    gaussian_noise_sd: Optional[float] = None,
    n_spikes: int = 40,
    n_short_pulses: int = 20,
    n_flatlines: int = 6,
    n_range_outliers: int = 12,
) -> Tuple[pd.DataFrame, SemiSyntheticTruth]:
    """
    构造半合成数据集。

    为什么必须做这个：
        真实胃肠压力数据没有绝对 ground truth，审稿人会质疑：
        “你怎么知道去噪后的信号更接近真实生理信号？”

    半合成策略：
        1. 从真实数据中提取一个相对干净的参考信号 clean_pressure；
        2. 保留真实生理事件结构；
        3. 人工注入已知位置和类型的伪影；
        4. 得到 observed_pressure；
        5. 后续算法输出可以和 clean_pressure / artifact_mask 真值比较。

    注入伪影类型：
        - isolated_spike       : 单点尖刺
        - short_pulse          : 短时脉冲
        - flatline_dropout     : 平台/丢包段
        - hard_range_outlier   : 超出生理范围的异常点

    返回：
        semi_df : 可直接输入你的 physiology_preserving_adaptive_denoise
        truth   : 半合成真值
    """
    rng = np.random.default_rng(random_state)

    semi_df = df.copy().reset_index(drop=True)
    t = semi_df[time_col].to_numpy(dtype=float)
    dt = estimate_dt(t)

    clean, event_mask = _infer_clean_reference(
        semi_df, time_col, pressure_col, event_col, baseline_col
    )

    n = len(clean)
    observed = clean.copy()

    if gaussian_noise_sd is None:
        # 用一阶差分估计一个温和噪声水平。
        gaussian_noise_sd = 0.35 * robust_mad(np.diff(clean))

    observed += rng.normal(0.0, gaussian_noise_sd, size=n)

    artifact_mask = np.zeros(n, dtype=bool)
    rows = []

    # 为避免把伪影大量注入到生理事件核心区，这里优先选择非事件区域。
    candidate_idx = np.where(~event_mask)[0]
    candidate_idx = candidate_idx[(candidate_idx > 5) & (candidate_idx < n - 6)]
    if len(candidate_idx) == 0:
        candidate_idx = np.arange(5, n - 6)

    def add_row(kind: str, s: int, e: int, amp: float):
        rows.append({
            "ArtifactType": kind,
            "StartIndex": int(s),
            "EndIndex": int(e - 1),
            "StartTime": float(t[s]),
            "EndTime": float(t[e - 1]),
            "Duration_s": float(t[e - 1] - t[s] + dt),
            "Amplitude": float(amp),
            "NumPoints": int(e - s),
        })

    # 1) 单点尖刺：模拟传感器瞬时跳变。
    for _ in range(n_spikes):
        i = int(rng.choice(candidate_idx))
        amp = float(rng.choice([-1.0, 1.0]) * rng.uniform(4.0, 12.0))
        observed[i] += amp
        artifact_mask[i] = True
        add_row("isolated_spike", i, i + 1, amp)

    # 2) 短脉冲：模拟短时机械冲击或局部压力假峰。
    max_pulse_len = max(2, int(round(4.8 / dt)))
    for _ in range(n_short_pulses):
        s = int(rng.choice(candidate_idx))
        length = int(rng.integers(2, max_pulse_len + 1))
        e = min(n, s + length)
        amp = float(rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 9.0))
        # 用半正弦形状，使短脉冲更接近真实形态伪影，而不是方波。
        shape = np.sin(np.linspace(0, np.pi, e - s))
        observed[s:e] += amp * shape
        artifact_mask[s:e] = True
        add_row("short_pulse", s, e, amp)

    # 3) flatline/dropout：模拟数据短时间卡住不变。
    max_flat_len = max(3, int(round(18.0 / dt)))
    min_flat_len = max(2, int(round(8.0 / dt)))
    for _ in range(n_flatlines):
        s = int(rng.choice(candidate_idx))
        length = int(rng.integers(min_flat_len, max_flat_len + 1))
        e = min(n, s + length)
        observed[s:e] = observed[s]
        artifact_mask[s:e] = True
        add_row("flatline_dropout", s, e, 0.0)

    # 4) hard range outlier：模拟明显不可能的压力值。
    for _ in range(n_range_outliers):
        i = int(rng.choice(candidate_idx))
        value = float(rng.choice([rng.uniform(60, 82), rng.uniform(132, 155)]))
        amp = value - observed[i]
        observed[i] = value
        artifact_mask[i] = True
        add_row("hard_range_outlier", i, i + 1, amp)

    semi_df[pressure_col] = observed
    if ph_col not in semi_df.columns:
        semi_df[ph_col] = np.nan
    if temp_col not in semi_df.columns:
        semi_df[temp_col] = np.nan

    truth = SemiSyntheticTruth(
        clean_pressure=clean,
        observed_pressure=observed,
        artifact_mask=artifact_mask,
        event_mask=event_mask,
        artifact_table=pd.DataFrame(rows),
    )
    return semi_df, truth


# ============================================================
# 3. 新增评价指标
# ============================================================

def compute_artifact_detection_metrics(
    pred_artifact_mask: np.ndarray,
    true_artifact_mask: np.ndarray,
) -> pd.DataFrame:
    """
    伪影检测指标。

    这些指标用于回答审稿人：
        “你的 ArtifactMask 真的找到了伪影吗？”

    指标解释：
        Precision : 预测为伪影的点中，有多少是真的伪影；
        Recall    : 真实伪影中，有多少被找出来；
        F1        : Precision 与 Recall 的调和平均；
        IoU       : 预测伪影区域与真实伪影区域的重叠程度。
    """
    pred = np.asarray(pred_artifact_mask, dtype=bool)
    true = np.asarray(true_artifact_mask, dtype=bool)

    if len(pred) != len(true):
        raise ValueError("pred_artifact_mask 和 true_artifact_mask 长度必须一致。")

    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    tn = np.logical_and(~pred, ~true).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)

    return pd.DataFrame([{
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "IoU": float(iou),
    }])


def compute_event_timing_metrics(
    pred_event_mask: np.ndarray,
    true_event_mask: np.ndarray,
    time: np.ndarray,
) -> pd.DataFrame:
    """
    计算事件起止时间误差。

    为什么重要：
        医学压力事件不仅幅值重要，发生时间也重要。
        如果滤波导致事件提前/滞后，可能影响胃肠转运时间、段落定位、动力学解释。

    匹配方式：
        对每个真实事件，找到 IoU 最大的预测事件作为匹配。
    """
    pred = np.asarray(pred_event_mask, dtype=bool)
    true = np.asarray(true_event_mask, dtype=bool)
    t = np.asarray(time, dtype=float)

    pred_comps = components(pred)
    true_comps = components(true)

    rows = []
    for event_id, (ts, te) in enumerate(true_comps, start=1):
        true_mask = np.zeros(len(t), dtype=bool)
        true_mask[ts:te] = True

        best = None
        best_iou = -1.0
        for ps, pe in pred_comps:
            pred_mask = np.zeros(len(t), dtype=bool)
            pred_mask[ps:pe] = True
            iou = mask_iou(true_mask, pred_mask)
            if np.isfinite(iou) and iou > best_iou:
                best_iou = iou
                best = (ps, pe)

        if best is None or best_iou <= 0:
            rows.append({
                "EventID": event_id,
                "Matched": False,
                "IoU": 0.0,
                "OnsetError_s": np.nan,
                "OffsetError_s": np.nan,
                "DurationError_s": np.nan,
            })
            continue

        ps, pe = best
        true_start = t[ts]
        true_end = t[te - 1]
        pred_start = t[ps]
        pred_end = t[pe - 1]

        rows.append({
            "EventID": event_id,
            "Matched": True,
            "IoU": float(best_iou),
            "OnsetError_s": float(pred_start - true_start),
            "OffsetError_s": float(pred_end - true_end),
            "DurationError_s": float((pred_end - pred_start) - (true_end - true_start)),
        })

    return pd.DataFrame(rows)


def compute_clean_signal_metrics(
    predicted_clean: np.ndarray,
    true_clean: np.ndarray,
    event_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    半合成数据上的 clean signal 重构误差。

    指标：
        RMSE / MAE / Corr：整体重构质量；
        Event_RMSE / NonEvent_RMSE：区分事件区和非事件区，证明算法没有牺牲生理事件。
    """
    pred = np.asarray(predicted_clean, dtype=float)
    true = np.asarray(true_clean, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(true)

    if valid.sum() < 3:
        raise ValueError("有效点太少，无法计算 clean signal metrics。")

    err = pred[valid] - true[valid]
    row = {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "Correlation": safe_corr(pred[valid], true[valid]),
    }

    if event_mask is not None:
        em = np.asarray(event_mask, dtype=bool)[valid]
        if em.sum() > 0:
            eerr = err[em]
            row["Event_RMSE"] = float(np.sqrt(np.mean(eerr ** 2)))
        else:
            row["Event_RMSE"] = np.nan

        if (~em).sum() > 0:
            neerr = err[~em]
            row["NonEvent_RMSE"] = float(np.sqrt(np.mean(neerr ** 2)))
        else:
            row["NonEvent_RMSE"] = np.nan

    return pd.DataFrame([row])


def evaluate_semisynthetic_result(
    pred_df: pd.DataFrame,
    truth: SemiSyntheticTruth,
    time_col: str = "Time",
    clean_col: str = "Pressure_Clean",
    artifact_col: str = "ArtifactMask",
    event_col: str = "PhysioEventMask",
) -> Dict[str, pd.DataFrame]:
    """
    对半合成实验结果进行一键评价。

    返回字典包含：
        clean_signal      : 与真值 clean_pressure 的重构误差；
        artifact_detection: 伪影检测 Precision / Recall / F1；
        event_timing      : 生理事件起止时间误差；
        summary           : 关键指标汇总，适合写论文表格。
    """
    pred_clean = pred_df[clean_col].to_numpy(dtype=float)
    pred_artifact = pred_df[artifact_col].to_numpy(dtype=bool)
    pred_event = pred_df[event_col].to_numpy(dtype=bool)
    time = pred_df[time_col].to_numpy(dtype=float)

    # 注意：如果算法重采样，pred_df 长度可能和 truth 不一致。
    # 这里将 truth 插值/投影到 pred_df 的时间轴上。
    # 假设 truth 来自同一时间范围，且原时间在 pred_df 中同名列可对齐。
    # 如果你的重采样 dt 与输入不同，也可以用这个逻辑保持公平比较。
    true_time = np.linspace(time[0], time[-1], len(truth.clean_pressure))
    true_clean = np.interp(time, true_time, truth.clean_pressure)

    true_artifact = np.zeros(len(time), dtype=bool)
    true_event = np.zeros(len(time), dtype=bool)
    for s, e in components(truth.artifact_mask):
        left = true_time[s]
        right = true_time[e - 1]
        true_artifact |= (time >= left) & (time <= right)
    for s, e in components(truth.event_mask):
        left = true_time[s]
        right = true_time[e - 1]
        true_event |= (time >= left) & (time <= right)

    clean_metrics = compute_clean_signal_metrics(pred_clean, true_clean, true_event)
    artifact_metrics = compute_artifact_detection_metrics(pred_artifact, true_artifact)
    timing_metrics = compute_event_timing_metrics(pred_event, true_event, time)

    summary = pd.DataFrame([{
        "RMSE": clean_metrics["RMSE"].iloc[0],
        "Correlation": clean_metrics["Correlation"].iloc[0],
        "Artifact_F1": artifact_metrics["F1"].iloc[0],
        "Artifact_Recall": artifact_metrics["Recall"].iloc[0],
        "Mean_Event_IoU": timing_metrics["IoU"].mean() if len(timing_metrics) else np.nan,
        "Mean_Abs_OnsetError_s": timing_metrics["OnsetError_s"].abs().mean() if len(timing_metrics) else np.nan,
        "Mean_Abs_OffsetError_s": timing_metrics["OffsetError_s"].abs().mean() if len(timing_metrics) else np.nan,
    }])

    return {
        "clean_signal": clean_metrics,
        "artifact_detection": artifact_metrics,
        "event_timing": timing_metrics,
        "summary": summary,
    }


# ============================================================
# 4. 消融实验框架
# ============================================================

def run_ablation_study(
    data: pd.DataFrame,
    denoise_func,
    base_kwargs: Dict[str, Any],
    truth: Optional[SemiSyntheticTruth] = None,
    time_col: str = "Time",
) -> pd.DataFrame:
    """
    消融实验框架。

    审稿人关心的问题：
        每个模块是否真的有贡献？

    这里设计 5 个版本：
        full_model                  : 完整模型；
        no_hard_range_gate          : 去掉生理范围硬门控；
        weak_physio_protection      : 弱化生理事件保护；
        weak_artifact_detector      : 弱化形态学伪影检测；
        no_final_smoothing          : 去掉最终事件保护型平滑。

    参数：
        data         : 输入数据
        denoise_func : 你的 physiology_preserving_adaptive_denoise 函数
        base_kwargs  : 完整模型参数
        truth        : 如果是半合成实验，传入 truth 后可计算客观指标

    返回：
        ablation_summary，每行一个模型变体。
    """
    variants = {}

    # 1. 完整模型
    variants["full_model"] = dict(base_kwargs)

    # 2. 不使用 hard range gate：验证生理范围先验是否有效。
    kw = dict(base_kwargs)
    kw["use_hard_range_gate"] = False
    variants["no_hard_range_gate"] = kw

    # 3. 弱化生理事件保护：提高事件检测门槛，使更多真实事件可能被当作普通波动处理。
    kw = dict(base_kwargs)
    kw["event_amp_k"] = kw.get("event_amp_k", 3.0) * 1.8
    kw["event_area_k"] = kw.get("event_area_k", 1.5) * 1.8
    variants["weak_physio_protection"] = kw

    # 4. 弱化伪影检测：提高 amp/slope/curvature 阈值，模拟不敏感的伪影检测器。
    kw = dict(base_kwargs)
    kw["amp_k"] = kw.get("amp_k", 6.0) * 1.8
    kw["slope_k"] = kw.get("slope_k", 6.0) * 1.8
    kw["curvature_k"] = kw.get("curvature_k", 6.0) * 1.8
    variants["weak_artifact_detector"] = kw

    # 5. 近似去掉最终平滑：将窗口压到非常小。
    kw = dict(base_kwargs)
    kw["smooth_window_s"] = max(0.1, estimate_dt(data[time_col].to_numpy(dtype=float)))
    variants["no_final_smoothing"] = kw

    rows = []
    for name, kw in variants.items():
        pred = denoise_func(data, **kw)

        row = {"Variant": name}

        # 基础内部指标：不依赖 ground truth。
        if "Pressure_Raw" in pred.columns and "Pressure_Clean" in pred.columns:
            raw = pred["Pressure_Raw"].to_numpy(dtype=float)
            clean = pred["Pressure_Clean"].to_numpy(dtype=float)
            row["FirstDiffEnergy_Raw"] = float(np.nansum(np.diff(raw) ** 2))
            row["FirstDiffEnergy_Clean"] = float(np.nansum(np.diff(clean) ** 2))
            row["FirstDiffEnergy_Reduction_%"] = float(
                100.0 * (row["FirstDiffEnergy_Raw"] - row["FirstDiffEnergy_Clean"])
                / (row["FirstDiffEnergy_Raw"] + 1e-9)
            )

        if "ArtifactMask" in pred.columns:
            row["ArtifactRatio"] = float(np.mean(pred["ArtifactMask"].to_numpy(dtype=bool)))

        if "PhysioEventMask" in pred.columns:
            row["PhysioEventRatio"] = float(np.mean(pred["PhysioEventMask"].to_numpy(dtype=bool)))

        # 半合成真值指标。
        if truth is not None:
            report = evaluate_semisynthetic_result(pred, truth)
            for col, val in report["summary"].iloc[0].items():
                row[col] = val

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 5. 事件保护型 VMD 模态选择
# ============================================================

@dataclass
class VMDModeSelectionResult:
    """
    VMD 模态选择结果。
    """
    modes: np.ndarray
    omega: np.ndarray
    mode_table: pd.DataFrame
    selected_modes: List[int]
    reconstructed: np.ndarray


def _run_vmd(signal: np.ndarray, alpha: float, tau: float, K: int, DC: int, init: int, tol: float):
    """
    延迟导入 vmdpy，避免没有安装时影响其他模块使用。
    """
    try:
        from vmdpy.vmdpy import VMD
    except Exception as exc:
        raise ImportError(
            "需要安装 vmdpy 才能运行 VMD：pip install vmdpy。"
        ) from exc

    return VMD(signal, alpha, tau, K, DC, init, tol)


def event_preserving_vmd_denoise(
    signal: np.ndarray,
    time: np.ndarray,
    event_mask: Optional[np.ndarray] = None,
    artifact_mask: Optional[np.ndarray] = None,
    K: int = 5,
    alpha: float = 2700,
    tau: float = 0.0,
    DC: int = 1,
    init: int = 1,
    tol: float = 1e-7,
    min_keep_score: float = 0.15,
    max_artifact_score: float = 0.65,
) -> VMDModeSelectionResult:
    """
    事件保护型 VMD 去噪。

    你原来的 VMD 用法更像：
        分解 -> 画每个 mode -> 人工判断。

    这里升级为论文更容易接受的自动选择：
        对每个 VMD mode 计算生理事件相关性、能量比例、伪影重叠能量、高频粗糙度，
        然后自动决定保留哪些 mode。

    核心思想：
        一个值得保留的 mode 应该满足：
        1. 在生理事件区有贡献；
        2. 与原信号或事件残差相关；
        3. 不主要集中在伪影区；
        4. 不只是高频抖动。

    参数：
        signal        : 输入压力信号，通常建议用 Pressure_Preclean
        time          : 时间轴
        event_mask    : 生理事件 mask。没有时仍可运行，但事件保护能力下降
        artifact_mask : 伪影 mask。没有时不会计算 artifact penalty
        K, alpha...   : VMD 参数

    返回：
        VMDModeSelectionResult
    """
    x = np.asarray(signal, dtype=float)
    t = np.asarray(time, dtype=float)

    valid = np.isfinite(x) & np.isfinite(t)
    if valid.sum() < 10:
        raise ValueError("有效点太少，无法运行 VMD。")

    # VMD 不接受 NaN，所以先用插值补齐。
    x_filled = x.copy()
    if not np.all(valid):
        x_filled[~valid] = np.interp(t[~valid], t[valid], x[valid])

    # 去均值可以让 VMD 更聚焦于动态成分；最终重构时再加回均值。
    x_mean = np.mean(x_filled)
    x_centered = x_filled - x_mean

    u, u_hat, omega = _run_vmd(x_centered, alpha, tau, K, DC, init, tol)

    # 对齐长度，vmdpy 某些情况下输出长度可能略有变化。
    n = min(len(x_centered), u.shape[1])
    x_centered = x_centered[:n]
    t = t[:n]
    u = u[:, :n]

    if event_mask is None:
        event_mask_arr = np.zeros(n, dtype=bool)
    else:
        event_mask_arr = np.asarray(event_mask, dtype=bool)[:n]

    if artifact_mask is None:
        artifact_mask_arr = np.zeros(n, dtype=bool)
    else:
        artifact_mask_arr = np.asarray(artifact_mask, dtype=bool)[:n]

    dt = estimate_dt(t)
    fs = 1.0 / dt
    freqs = fftfreq(n, d=dt)
    pos_freqs = freqs[: n // 2]

    total_energy = np.sum(x_centered ** 2) + 1e-9
    rows = []
    keep_modes = []

    # baseline 用于得到事件残差，让 mode 是否承载事件形态更容易判断。
    baseline = rolling_median(x_centered, max(9, int(round(60.0 / dt))))
    event_residual = x_centered - baseline

    for k in range(K):
        mode = u[k]
        energy = np.sum(mode ** 2)
        energy_ratio = energy / total_energy

        # mode 与整体信号的相关性。
        signal_corr = safe_corr(mode, x_centered)

        # mode 与事件残差的相关性：越高说明越可能承载真实生理事件。
        if event_mask_arr.sum() >= 3:
            event_corr = safe_corr(mode[event_mask_arr], event_residual[event_mask_arr])
            event_energy_ratio = np.sum(mode[event_mask_arr] ** 2) / (energy + 1e-9)
        else:
            event_corr = 0.0
            event_energy_ratio = 0.0

        # mode 在伪影区的能量占比：越高越可疑。
        if artifact_mask_arr.sum() >= 1:
            artifact_energy_ratio = np.sum(mode[artifact_mask_arr] ** 2) / (energy + 1e-9)
        else:
            artifact_energy_ratio = 0.0

        # 高频粗糙度：一阶差分能量 / mode 能量。
        roughness = np.sum(np.diff(mode) ** 2) / (energy + 1e-9)

        # 频谱主频，用于论文解释每个模态的频域属性。
        spec = np.abs(fft(mode))[: n // 2]
        if len(spec) > 0 and np.max(spec) > 0:
            dominant_freq = float(pos_freqs[np.argmax(spec)])
        else:
            dominant_freq = np.nan

        # 归一化高频惩罚。
        # 这里用 roughness 的简单压缩形式，避免某个极端 mode 使 score 爆炸。
        roughness_penalty = roughness / (roughness + 1.0)

        # 事件保护型保留分数：
        #   正项：整体相关性、事件相关性、事件区能量；
        #   负项：伪影区能量、高频粗糙度。
        keep_score = (
            0.35 * max(signal_corr if np.isfinite(signal_corr) else 0.0, 0.0)
            + 0.30 * max(event_corr if np.isfinite(event_corr) else 0.0, 0.0)
            + 0.20 * event_energy_ratio
            + 0.15 * energy_ratio
            - 0.35 * artifact_energy_ratio
            - 0.15 * roughness_penalty
        )

        # 保留规则：
        #   1. 分数足够高；
        #   2. 不能主要集中在伪影区；
        #   3. 如果是 DC/低频 mode，通常保留，因为它承载 baseline。
        keep = (keep_score >= min_keep_score) and (artifact_energy_ratio <= max_artifact_score)
        if DC == 1 and k == 0:
            keep = True

        if keep:
            keep_modes.append(k)

        rows.append({
            "Mode": k + 1,
            "ModeIndex0": k,
            "CenterOmega_Last": float(omega[-1, k]) if np.ndim(omega) == 2 else np.nan,
            "DominantFreq_Hz": dominant_freq,
            "EnergyRatio": float(energy_ratio),
            "SignalCorrelation": float(signal_corr) if np.isfinite(signal_corr) else np.nan,
            "EventCorrelation": float(event_corr) if np.isfinite(event_corr) else np.nan,
            "EventEnergyRatio": float(event_energy_ratio),
            "ArtifactEnergyRatio": float(artifact_energy_ratio),
            "Roughness": float(roughness),
            "KeepScore": float(keep_score),
            "Selected": bool(keep),
        })

    # 如果规则过严导致没有 mode 被保留，则至少保留信号相关性最高的 mode。
    if len(keep_modes) == 0:
        table_tmp = pd.DataFrame(rows)
        best = int(table_tmp["SignalCorrelation"].fillna(-np.inf).idxmax())
        keep_modes = [best]
        rows[best]["Selected"] = True

    reconstructed = np.sum(u[keep_modes, :], axis=0) + x_mean

    mode_table = pd.DataFrame(rows)

    return VMDModeSelectionResult(
        modes=u,
        omega=omega,
        mode_table=mode_table,
        selected_modes=keep_modes,
        reconstructed=reconstructed,
    )


# ============================================================
# 6. 方便论文画表的汇总函数
# ============================================================

def summarize_vmd_result(vmd_result: VMDModeSelectionResult) -> pd.DataFrame:
    """
    提取 VMD 模态选择结果，生成适合论文表格的 summary。
    """
    table = vmd_result.mode_table.copy()
    cols = [
        "Mode", "DominantFreq_Hz", "EnergyRatio", "SignalCorrelation",
        "EventCorrelation", "EventEnergyRatio", "ArtifactEnergyRatio",
        "KeepScore", "Selected"
    ]
    return table[cols]
