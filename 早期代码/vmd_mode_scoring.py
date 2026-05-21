# -*- coding: utf-8 -*-
"""
vmd_mode_scoring.py

Physiology-constrained VMD mode scoring and reconstruction.

目的：
    将普通 VMD 后的“人工选模态”升级为可解释、可复现的
    生理约束型 VMD 模态评分系统。

核心思想：
    一个应该被保留的 VMD mode 通常应满足：
    1) 与原始压力动态相关；
    2) 在生理事件区域有足够贡献；
    3) 不主要集中在伪影区域；
    4) 频谱不过度随机化；
    5) 时间形态具有连续性，而不是尖刺/抖动主导。

建议输入：
    signal        : Pressure_Preclean 或 Pressure_Clean 之前的预清洗压力；
    time          : Time；
    event_mask    : PhysioEventMask；
    artifact_mask : ArtifactMask。

依赖：
    numpy, pandas, scipy, vmdpy

安装 VMD：
    pip install vmdpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis


# ============================================================
# 1. 基础工具函数
# ============================================================

def estimate_dt(time: np.ndarray) -> float:
    """
    使用中位数估计采样间隔，避免少量异常时间间隔影响采样率估计。
    """
    t = np.asarray(time, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("time 至少需要两个递增采样点。")
    return float(np.median(dt))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    安全计算 Pearson 相关系数。

    如果有效点过少或任一数组近似常数，返回 np.nan，避免后续评分被异常值污染。
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


def robust_minmax01(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    将一组 mode-level 特征稳健归一化到 0~1。

    为什么不用普通 min-max：
        某个极端 mode 可能导致其他 mode 被压得过低。
        这里用 5% 和 95% 分位数裁剪，增强稳定性。
    """
    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return out

    lo = np.nanpercentile(v[finite], 5)
    hi = np.nanpercentile(v[finite], 95)
    if abs(hi - lo) < eps:
        out[finite] = 0.5
        return out

    out[finite] = np.clip((v[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def rolling_median_np(x: np.ndarray, window: int) -> np.ndarray:
    """
    简单 rolling median，用于估计 VMD 输入信号的慢变化 baseline。
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


def fill_nan_by_interp(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """
    VMD 不能处理 NaN，因此用线性插值填补缺失值。
    """
    t = np.asarray(time, dtype=float)
    x = np.asarray(signal, dtype=float).copy()
    valid = np.isfinite(t) & np.isfinite(x)
    if valid.sum() < 2:
        raise ValueError("有效信号点少于 2 个，无法插值补齐。")
    if not valid.all():
        x[~valid] = np.interp(t[~valid], t[valid], x[valid])
    return x


# ============================================================
# 2. 单个 mode 的特征
# ============================================================

def compute_spectral_entropy(mode: np.ndarray, eps: float = 1e-12) -> float:
    """
    计算归一化频谱熵，范围约为 0~1。

    解释：
        - 频谱集中：entropy 低，常见于趋势或规则生理波动；
        - 频谱分散：entropy 高，常见于随机噪声或复杂伪影。
    """
    x = np.asarray(mode, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan

    power = np.abs(fft(x))[: len(x) // 2] ** 2
    power = power[np.isfinite(power)]
    if len(power) == 0 or np.sum(power) <= eps:
        return 0.0

    p = power / (np.sum(power) + eps)
    ent = -np.sum(p * np.log(p + eps))
    ent_norm = ent / (np.log(len(p)) + eps)
    return float(np.clip(ent_norm, 0.0, 1.0))


def compute_mode_kurtosis(mode: np.ndarray) -> float:
    """
    计算 mode 的峰度。

    解释：
        spike-like 伪影通常具有较高峰度。
        这里使用 Fisher=False，使高斯分布峰度约为 3。
    """
    x = np.asarray(mode, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4 or np.std(x) < 1e-12:
        return np.nan
    return float(kurtosis(x, fisher=False, bias=False))


def compute_temporal_continuity(mode: np.ndarray, eps: float = 1e-12) -> float:
    """
    计算时间连续性评分，范围约为 0~1，越高越连续。

    思路：
        一阶差分能量 / mode 能量 越大，说明越粗糙；
        将粗糙度转换成 continuity = 1 / (1 + roughness)。
    """
    x = np.asarray(mode, dtype=float)
    valid = np.isfinite(x)
    x = x[valid]
    if len(x) < 3:
        return np.nan

    energy = np.sum(x ** 2) + eps
    roughness = np.sum(np.diff(x) ** 2) / energy
    return float(1.0 / (1.0 + roughness))


def compute_mask_energy_ratio(mode: np.ndarray, mask: Optional[np.ndarray], eps: float = 1e-12) -> float:
    """
    计算 mode 能量有多少比例落在给定 mask 区域。
    """
    x = np.asarray(mode, dtype=float)
    energy = np.sum(x ** 2) + eps
    if mask is None:
        return 0.0

    m = np.asarray(mask, dtype=bool)[: len(x)]
    if m.sum() == 0:
        return 0.0

    return float(np.sum(x[m] ** 2) / energy)


def compute_dominant_frequency(mode: np.ndarray, dt: float) -> float:
    """
    计算 mode 的主频，单位 Hz。
    """
    x = np.asarray(mode, dtype=float)
    n = len(x)
    if n < 4:
        return np.nan

    freqs = fftfreq(n, d=dt)[: n // 2]
    amp = np.abs(fft(x))[: n // 2]
    if len(amp) == 0 or np.max(amp) <= 0:
        return np.nan
    return float(freqs[np.argmax(amp)])


# ============================================================
# 3. VMD 运行与结果结构
# ============================================================

@dataclass
class PhysiologyConstrainedVMDResult:
    """
    生理约束型 VMD 重构结果。
    """
    modes: np.ndarray
    omega: np.ndarray
    mode_table: pd.DataFrame
    selected_modes: List[int]
    reconstructed: np.ndarray
    reconstructed_centered: np.ndarray
    signal_mean: float
    weights: Dict[str, float]


def run_vmd(signal_centered: np.ndarray, alpha: float, tau: float, K: int, DC: int, init: int, tol: float):
    """
    运行 vmdpy.VMD。
    """
    try:
        from vmdpy.vmdpy import VMD
    except Exception as exc:
        raise ImportError("需要安装 vmdpy：pip install vmdpy") from exc

    return VMD(signal_centered, alpha, tau, K, DC, init, tol)


# ============================================================
# 4. mode 特征表与评分
# ============================================================

def extract_vmd_mode_features(
    modes: np.ndarray,
    signal_centered: np.ndarray,
    time: np.ndarray,
    event_mask: Optional[np.ndarray] = None,
    artifact_mask: Optional[np.ndarray] = None,
    baseline_window_s: float = 60.0,
) -> pd.DataFrame:
    """
    为每个 VMD mode 计算可解释特征。

    输出字段适合直接写入论文表格或补充材料。
    """
    u = np.asarray(modes, dtype=float)
    x = np.asarray(signal_centered, dtype=float)
    t = np.asarray(time, dtype=float)

    n = min(u.shape[1], len(x), len(t))
    u = u[:, :n]
    x = x[:n]
    t = t[:n]

    dt = estimate_dt(t)
    baseline_w = max(9, int(round(baseline_window_s / dt)))
    if baseline_w % 2 == 0:
        baseline_w += 1
    baseline = rolling_median_np(x, baseline_w)
    event_residual = x - baseline

    e_mask = np.zeros(n, dtype=bool) if event_mask is None else np.asarray(event_mask, dtype=bool)[:n]
    a_mask = np.zeros(n, dtype=bool) if artifact_mask is None else np.asarray(artifact_mask, dtype=bool)[:n]

    total_energy = np.sum(x ** 2) + 1e-12
    rows = []

    for k in range(u.shape[0]):
        mode = u[k]
        mode_energy = np.sum(mode ** 2) + 1e-12

        event_corr = safe_corr(mode[e_mask], event_residual[e_mask]) if e_mask.sum() >= 3 else np.nan
        non_event_mask = ~e_mask
        non_event_corr = safe_corr(mode[non_event_mask], x[non_event_mask]) if non_event_mask.sum() >= 3 else np.nan

        rows.append({
            "Mode": k + 1,
            "ModeIndex0": k,
            "DominantFreq_Hz": compute_dominant_frequency(mode, dt),
            "EnergyRatio": float(mode_energy / total_energy),
            "SignalCorrelation": safe_corr(mode, x),
            "EventCorrelation": event_corr,
            "NonEventCorrelation": non_event_corr,
            "EventEnergyRatio": compute_mask_energy_ratio(mode, e_mask),
            "ArtifactEnergyRatio": compute_mask_energy_ratio(mode, a_mask),
            "SpectralEntropy": compute_spectral_entropy(mode),
            "Kurtosis": compute_mode_kurtosis(mode),
            "TemporalContinuity": compute_temporal_continuity(mode),
        })

    return pd.DataFrame(rows)


def add_physiology_keep_score(
    mode_table: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    根据多指标为每个 mode 计算 PhysioKeepScore。

    默认权重解释：
        正向贡献：
            SignalCorrelation      mode 与整体信号相似；
            EventCorrelation       mode 与事件残差相似；
            EventEnergyRatio       mode 在事件区有贡献；
            TemporalContinuity     mode 时间上连续。

        负向惩罚：
            ArtifactEnergyRatio    mode 主要活跃在伪影区；
            SpectralEntropy        mode 频谱过于分散，偏随机；
            Kurtosis               mode 尖刺性强。
    """
    if weights is None:
        weights = {
            "signal_corr": 0.22,
            "event_corr": 0.26,
            "event_energy": 0.20,
            "continuity": 0.16,
            "artifact_penalty": 0.22,
            "entropy_penalty": 0.10,
            "kurtosis_penalty": 0.08,
        }

    df = mode_table.copy()

    # 相关性负值通常说明该 mode 与目标形态相反，作为 0 处理更稳健。
    signal_corr_pos = np.maximum(df["SignalCorrelation"].fillna(0.0).to_numpy(dtype=float), 0.0)
    event_corr_pos = np.maximum(df["EventCorrelation"].fillna(0.0).to_numpy(dtype=float), 0.0)

    event_energy = df["EventEnergyRatio"].fillna(0.0).to_numpy(dtype=float)
    artifact_energy = df["ArtifactEnergyRatio"].fillna(0.0).to_numpy(dtype=float)
    entropy = df["SpectralEntropy"].fillna(0.0).to_numpy(dtype=float)
    continuity = df["TemporalContinuity"].fillna(0.0).to_numpy(dtype=float)

    # 峰度可能跨度很大，因此先稳健归一化。
    kurt_penalty = robust_minmax01(df["Kurtosis"].to_numpy(dtype=float))

    score = (
        weights["signal_corr"] * signal_corr_pos
        + weights["event_corr"] * event_corr_pos
        + weights["event_energy"] * event_energy
        + weights["continuity"] * continuity
        - weights["artifact_penalty"] * artifact_energy
        - weights["entropy_penalty"] * entropy
        - weights["kurtosis_penalty"] * kurt_penalty
    )

    df["KurtosisPenalty01"] = kurt_penalty
    df["PhysioKeepScore"] = score
    df["WeightsUsed"] = str(weights)
    return df


def select_physiology_relevant_modes(
    scored_table: pd.DataFrame,
    min_keep_score: float = 0.18,
    max_artifact_energy_ratio: float = 0.65,
    always_keep_first_mode: bool = True,
    min_modes: int = 1,
    max_modes: Optional[int] = None,
) -> Tuple[List[int], pd.DataFrame]:
    """
    根据 PhysioKeepScore 自动选择 VMD modes。

    选择原则：
        1. 分数足够高；
        2. 不被伪影区能量主导；
        3. 默认保留第一个低频/DC mode，避免 baseline 丢失；
        4. 如果过严导致没有 mode 被选，则保留分数最高的 min_modes 个。
    """
    df = scored_table.copy()
    df["Selected"] = False

    candidate = (
        (df["PhysioKeepScore"] >= min_keep_score)
        & (df["ArtifactEnergyRatio"].fillna(0.0) <= max_artifact_energy_ratio)
    )

    selected = df.loc[candidate, "ModeIndex0"].astype(int).tolist()

    if always_keep_first_mode and 0 not in selected and len(df) > 0:
        selected.append(0)

    if max_modes is not None and len(selected) > max_modes:
        ranked = (
            df[df["ModeIndex0"].isin(selected)]
            .sort_values("PhysioKeepScore", ascending=False)
            .head(max_modes)
        )
        selected = ranked["ModeIndex0"].astype(int).tolist()
        if always_keep_first_mode and 0 not in selected and len(df) > 0:
            selected[-1] = 0

    if len(selected) < min_modes:
        ranked = df.sort_values("PhysioKeepScore", ascending=False).head(min_modes)
        selected = sorted(set(selected + ranked["ModeIndex0"].astype(int).tolist()))

    selected = sorted(set(selected))
    df.loc[df["ModeIndex0"].isin(selected), "Selected"] = True
    return selected, df


# ============================================================
# 5. 主函数：生理约束型 VMD 重构
# ============================================================

def physiology_constrained_vmd_reconstruction(
    signal: np.ndarray,
    time: np.ndarray,
    event_mask: Optional[np.ndarray] = None,
    artifact_mask: Optional[np.ndarray] = None,
    K: int = 5,
    alpha: float = 2700.0,
    tau: float = 0.0,
    DC: int = 1,
    init: int = 1,
    tol: float = 1e-7,
    min_keep_score: float = 0.18,
    max_artifact_energy_ratio: float = 0.65,
    always_keep_first_mode: bool = True,
    min_modes: int = 1,
    max_modes: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
) -> PhysiologyConstrainedVMDResult:
    """
    生理约束型 VMD 模态评分与重构。

    推荐用法：
        result = physiology_constrained_vmd_reconstruction(
            signal=pred_df["Pressure_Preclean"].values,
            time=pred_df["Time"].values,
            event_mask=pred_df["PhysioEventMask"].values,
            artifact_mask=pred_df["ArtifactMask"].values,
            K=5,
            alpha=2700,
        )

        result.mode_table
        result.reconstructed

    返回：
        PhysiologyConstrainedVMDResult
    """
    t = np.asarray(time, dtype=float)
    x = fill_nan_by_interp(t, signal)

    n = min(len(t), len(x))
    t = t[:n]
    x = x[:n]

    x_mean = float(np.mean(x))
    x_centered = x - x_mean

    modes, u_hat, omega = run_vmd(
        x_centered,
        alpha=alpha,
        tau=tau,
        K=K,
        DC=DC,
        init=init,
        tol=tol,
    )

    # 对齐长度，vmdpy 输出长度有时可能略短。
    n2 = min(len(x_centered), modes.shape[1])
    modes = modes[:, :n2]
    t = t[:n2]
    x_centered = x_centered[:n2]

    e_mask = None if event_mask is None else np.asarray(event_mask, dtype=bool)[:n2]
    a_mask = None if artifact_mask is None else np.asarray(artifact_mask, dtype=bool)[:n2]

    feature_table = extract_vmd_mode_features(
        modes=modes,
        signal_centered=x_centered,
        time=t,
        event_mask=e_mask,
        artifact_mask=a_mask,
    )

    scored_table = add_physiology_keep_score(feature_table, weights=weights)
    selected_modes, final_table = select_physiology_relevant_modes(
        scored_table,
        min_keep_score=min_keep_score,
        max_artifact_energy_ratio=max_artifact_energy_ratio,
        always_keep_first_mode=always_keep_first_mode,
        min_modes=min_modes,
        max_modes=max_modes,
    )

    reconstructed_centered = np.sum(modes[selected_modes, :], axis=0)
    reconstructed = reconstructed_centered + x_mean

    weights_used = weights if weights is not None else {
        "signal_corr": 0.22,
        "event_corr": 0.26,
        "event_energy": 0.20,
        "continuity": 0.16,
        "artifact_penalty": 0.22,
        "entropy_penalty": 0.10,
        "kurtosis_penalty": 0.08,
    }

    return PhysiologyConstrainedVMDResult(
        modes=modes,
        omega=omega,
        mode_table=final_table,
        selected_modes=selected_modes,
        reconstructed=reconstructed,
        reconstructed_centered=reconstructed_centered,
        signal_mean=x_mean,
        weights=weights_used,
    )


# ============================================================
# 6. 论文表格友好输出
# ============================================================

def summarize_physiology_vmd_result(result: PhysiologyConstrainedVMDResult) -> pd.DataFrame:
    """
    输出适合论文展示的 VMD mode summary。
    """
    cols = [
        "Mode",
        "DominantFreq_Hz",
        "EnergyRatio",
        "SignalCorrelation",
        "EventCorrelation",
        "EventEnergyRatio",
        "ArtifactEnergyRatio",
        "SpectralEntropy",
        "Kurtosis",
        "TemporalContinuity",
        "PhysioKeepScore",
        "Selected",
    ]
    return result.mode_table[cols].copy()


def compare_vmd_reconstruction_to_reference(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    event_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    对 VMD 重构信号进行简单评价。

    reference 可以是：
        - 半合成 truth.clean_pressure；
        - 或你的 Pressure_Clean / pseudo-clean reference。
    """
    y = np.asarray(reconstructed, dtype=float)
    r = np.asarray(reference, dtype=float)
    n = min(len(y), len(r))
    y = y[:n]
    r = r[:n]

    valid = np.isfinite(y) & np.isfinite(r)
    if valid.sum() < 3:
        raise ValueError("有效点太少，无法评价 VMD 重构。")

    err = y[valid] - r[valid]
    row = {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "Correlation": safe_corr(y[valid], r[valid]),
    }

    if event_mask is not None:
        m = np.asarray(event_mask, dtype=bool)[:n][valid]
        if m.sum() > 0:
            row["Event_RMSE"] = float(np.sqrt(np.mean(err[m] ** 2)))
        else:
            row["Event_RMSE"] = np.nan

        if (~m).sum() > 0:
            row["NonEvent_RMSE"] = float(np.sqrt(np.mean(err[~m] ** 2)))
        else:
            row["NonEvent_RMSE"] = np.nan

    return pd.DataFrame([row])
