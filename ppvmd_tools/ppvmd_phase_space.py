# -*- coding: utf-8 -*-
"""
ppvmd_phase_space.py

Phase-space reconstruction and nonlinear feature extraction module for
physiology-preserving VMD gastrointestinal pressure analysis.

中文说明：
    本文件用于你当前的研究主线：

        Raw pressure
        → physiology-preserving denoising
        → physiology-constrained VMD
        → event-guided phase-space reconstruction
        → nonlinear feature extraction
        → healthy vs patient classification

    该模块重点解决：
        1. 如何从 PhysioEventMask 中提取事件中心窗口；
        2. 如何构造固定窗口 1024-point baseline；
        3. 如何自动估计相空间重构参数 τ 和 m；
        4. 如何从原始 Clean 信号、VMD 重构信号、单个 VMD mode 中提取非线性动力学特征；
        5. 如何输出适合后续机器学习分类的 feature table。

SCI方法学建议：
    - 不建议只对每个短事件单独重构相空间，因为许多事件过短，吸引子估计不稳定。
    - 推荐使用 event-guided windows，即以生理事件中心为锚点，向前后扩展固定或自适应窗口。
    - 同时保留 fixed 1024-point windows 作为传统基线方法，便于和已有博士论文/文献对比。

依赖：
    numpy, pandas, scipy

可选：
    sklearn 不是必须；本模块只输出特征表，分类可在 notebook 中另做。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis


# ============================================================
# 0. 基础工具函数
# ============================================================


def estimate_dt(time: np.ndarray) -> float:
    """
    使用中位数估计采样间隔。

    为什么用中位数：
        胃肠胶囊数据可能存在偶发时间间隔异常；均值容易被异常间隔影响。
        中位数更稳健。
    """
    t = np.asarray(time, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("time 至少需要两个递增采样点。")
    return float(np.median(dt))


def robust_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    稳健 z-score 标准化。

    用 median 和 MAD，而不是 mean/std。
    这样可以降低尖峰伪影或极端事件对标准化的影响。
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    mad = max(float(mad), eps)
    return (x - med) / mad


def minmax_scale(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    0-1 标准化。主要用于可视化，不建议作为默认动力学特征输入。
    """
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    return (x - lo) / (hi - lo + eps)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    安全计算 Pearson 相关系数。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def get_segments(mask: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """
    获取布尔 mask 中连续 True 的区间。

    返回：
        [(start, end), ...]
    其中 end 为开区间，不包含 end。
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
        e = i
        if e - s >= min_len:
            out.append((s, e))
    return out


def align_length(*arrays: np.ndarray) -> List[np.ndarray]:
    """
    将多个数组裁剪到相同最短长度。
    """
    n = min(len(a) for a in arrays)
    return [np.asarray(a)[:n] for a in arrays]


# ============================================================
# 1. 窗口提取：固定窗口与事件引导窗口
# ============================================================


@dataclass
class WindowSpec:
    """
    相空间重构窗口的元信息。

    字段：
        window_id       : 窗口编号
        method          : 'fixed' 或 'event_guided'
        start_idx/end_idx: 窗口索引范围，end_idx 为开区间
        start_time/end_time: 窗口时间范围
        center_time     : 中心时间
        event_id        : 如果是事件引导窗口，对应事件编号；固定窗口则为 None
        event_start_time/event_end_time: 原始事件起止时间
        num_points      : 窗口内点数
        event_coverage  : 窗口中 PhysioEventMask=True 的比例
    """
    window_id: int
    method: str
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    center_time: float
    event_id: Optional[int]
    event_start_time: Optional[float]
    event_end_time: Optional[float]
    num_points: int
    event_coverage: float


def extract_fixed_windows(
    time: np.ndarray,
    event_mask: Optional[np.ndarray] = None,
    window_points: int = 1024,
    step_points: Optional[int] = None,
    min_valid_points: int = 256,
) -> List[WindowSpec]:
    """
    固定长度窗口提取。

    用途：
        作为传统基线方法，例如博士论文中常用 1024 个采样点为一组。

    参数：
        window_points : 每个窗口点数，例如 1024。
        step_points   : 滑动步长。None 时默认为 window_points，即无重叠。
    """
    t = np.asarray(time, dtype=float)
    n = len(t)
    if step_points is None:
        step_points = window_points

    if event_mask is None:
        em = np.zeros(n, dtype=bool)
    else:
        em = np.asarray(event_mask, dtype=bool)[:n]

    specs = []
    wid = 1
    for s in range(0, n - min_valid_points + 1, step_points):
        e = min(s + window_points, n)
        if e - s < min_valid_points:
            continue
        center = 0.5 * (t[s] + t[e - 1])
        specs.append(WindowSpec(
            window_id=wid,
            method="fixed",
            start_idx=s,
            end_idx=e,
            start_time=float(t[s]),
            end_time=float(t[e - 1]),
            center_time=float(center),
            event_id=None,
            event_start_time=None,
            event_end_time=None,
            num_points=int(e - s),
            event_coverage=float(em[s:e].mean()) if e > s else 0.0,
        ))
        wid += 1
    return specs


def extract_event_guided_windows(
    time: np.ndarray,
    event_mask: np.ndarray,
    pre_s: float = 180.0,
    post_s: float = 180.0,
    min_event_duration_s: float = 6.0,
    min_window_points: int = 128,
    merge_overlapping: bool = False,
    max_event_gap_s_for_merge: float = 0.0,
    fixed_center_window: bool = True,
) -> List[WindowSpec]:
    """
    事件引导窗口提取。

    推荐用于相空间重构的严谨版本：
        fixed_center_window=True 时，窗口为事件中心 ± pre_s/post_s。
        这样每个窗口长度基本一致，避免超长事件导致 NumPoints 过大。

    注意：
        为了避免不同事件窗口被合并成长窗口，默认 merge_overlapping=False。
    """
    t = np.asarray(time, dtype=float)
    em = np.asarray(event_mask, dtype=bool)[:len(t)]

    dt = estimate_dt(t)
    min_event_len = max(1, int(round(min_event_duration_s / dt)))
    comps = get_segments(em, min_len=min_event_len)

    raw_windows = []

    for eid, (s, e) in enumerate(comps, start=1):
        event_start = float(t[s])
        event_end = float(t[e - 1])
        center_time = 0.5 * (event_start + event_end)

        if fixed_center_window:
            left_time = center_time - pre_s
            right_time = center_time + post_s
        else:
            left_time = event_start - pre_s
            right_time = event_end + post_s

        start_idx = int(np.searchsorted(t, left_time, side="left"))
        end_idx = int(np.searchsorted(t, right_time, side="right"))

        start_idx = max(0, start_idx)
        end_idx = min(len(t), end_idx)

        if end_idx - start_idx < min_window_points:
            continue

        raw_windows.append({
            "event_ids": [eid],
            "event_start_idx": s,
            "event_end_idx": e,
            "event_start_time": event_start,
            "event_end_time": event_end,
            "center_time": center_time,
            "start_idx": start_idx,
            "end_idx": end_idx,
        })

    if not merge_overlapping:
        windows = raw_windows
    else:
        windows = []
        for w in raw_windows:
            if not windows:
                windows.append(w)
                continue

            prev = windows[-1]
            gap_s = t[w["start_idx"]] - t[prev["end_idx"] - 1]

            if w["start_idx"] <= prev["end_idx"] or gap_s <= max_event_gap_s_for_merge:
                prev["end_idx"] = max(prev["end_idx"], w["end_idx"])
                prev["event_ids"].extend(w["event_ids"])
                prev["event_end_idx"] = max(prev["event_end_idx"], w["event_end_idx"])
                prev["event_end_time"] = max(prev["event_end_time"], w["event_end_time"])
                prev["center_time"] = 0.5 * (
                    prev["event_start_time"] + prev["event_end_time"]
                )
            else:
                windows.append(w)

    specs = []

    for wid, w in enumerate(windows, start=1):
        s = w["start_idx"]
        e = w["end_idx"]

        event_id = w["event_ids"][0] if len(w["event_ids"]) == 1 else None

        specs.append(WindowSpec(
            window_id=wid,
            method="event_guided",
            start_idx=s,
            end_idx=e,
            start_time=float(t[s]),
            end_time=float(t[e - 1]),
            center_time=float(w["center_time"]),
            event_id=event_id,
            event_start_time=w["event_start_time"],
            event_end_time=w["event_end_time"],
            num_points=int(e - s),
            event_coverage=float(em[s:e].mean()) if e > s else 0.0,
        ))

    return specs


def window_specs_to_dataframe(windows: List[WindowSpec]) -> pd.DataFrame:
    """
    将 WindowSpec 列表转换成 DataFrame，便于查看和保存。
    """
    return pd.DataFrame([w.__dict__ for w in windows])


# ============================================================
# 2. 相空间重构参数估计：tau 与 m
# ============================================================


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    计算自相关函数 ACF。

    用途：
        估计 delay time tau。
        常用规则：取 ACF 第一次低于 1/e 或第一次过零的位置。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max_lag + 2:
        max_lag = max(1, len(x) - 2)
    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return np.full(max_lag + 1, np.nan)
    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x[:-lag], x[lag:]) / ((len(x) - lag) * var)
    return acf


def estimate_tau_autocorr(
    x: np.ndarray,
    max_lag: int = 200,
    threshold: float = 1 / np.e,
    min_tau: int = 1,
) -> int:
    """
    使用自相关 1/e 原则估计 tau。

    若没有低于 threshold，则返回第一个局部极小值；仍没有则返回 min_tau。
    """
    acf = autocorrelation(x, max_lag=max_lag)
    for lag in range(max(min_tau, 1), len(acf)):
        if np.isfinite(acf[lag]) and acf[lag] <= threshold:
            return int(lag)
    # fallback: 第一个局部极小值
    valid = np.isfinite(acf)
    if valid.sum() > 3:
        peaks, _ = find_peaks(-acf[1:])
        if len(peaks) > 0:
            return int(peaks[0] + 1)
    return int(min_tau)


def _hist_entropy(values: np.ndarray, bins: int) -> float:
    """
    离散熵，用于 AMI 估计。
    """
    hist, _ = np.histogram(values, bins=bins)
    p = hist.astype(float)
    p = p[p > 0]
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p + 1e-12)))


def average_mutual_information(
    x: np.ndarray,
    max_lag: int = 200,
    bins: Union[int, str] = "fd",
) -> np.ndarray:
    """
    估计 Average Mutual Information, AMI。

    AMI 用于选择 delay time tau：
        常用规则是取 AMI 的第一个局部极小值。

    bins:
        'fd' 表示使用 Freedman-Diaconis 规则自动估计 bins。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max_lag + 5:
        max_lag = max(1, len(x) - 5)

    if bins == "fd":
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        bw = 2 * iqr / (len(x) ** (1 / 3) + 1e-12)
        if bw <= 1e-12:
            nbins = int(np.sqrt(len(x)))
        else:
            nbins = int(np.ceil((np.max(x) - np.min(x)) / bw))
        nbins = int(np.clip(nbins, 8, 64))
    else:
        nbins = int(bins)

    ami = np.full(max_lag + 1, np.nan, dtype=float)
    ami[0] = _hist_entropy(x, nbins)
    for lag in range(1, max_lag + 1):
        x1 = x[:-lag]
        x2 = x[lag:]
        if len(x1) < 10:
            break
        h1 = _hist_entropy(x1, nbins)
        h2 = _hist_entropy(x2, nbins)
        h12, _, _ = np.histogram2d(x1, x2, bins=nbins)
        p12 = h12.astype(float)
        p12 = p12[p12 > 0]
        p12 = p12 / np.sum(p12)
        joint_entropy = -np.sum(p12 * np.log(p12 + 1e-12))
        ami[lag] = h1 + h2 - joint_entropy
    return ami


def estimate_tau_ami(
    x: np.ndarray,
    max_lag: int = 200,
    bins: Union[int, str] = "fd",
    min_tau: int = 1,
) -> int:
    """
    使用 AMI 第一个局部极小值估计 tau。

    如果 AMI 无有效局部极小值，则 fallback 到 autocorrelation 方法。
    """
    ami = average_mutual_information(x, max_lag=max_lag, bins=bins)
    valid = np.isfinite(ami)
    if valid.sum() > 5:
        inv = -ami
        peaks, _ = find_peaks(inv[min_tau:])
        if len(peaks) > 0:
            return int(peaks[0] + min_tau)
    return estimate_tau_autocorr(x, max_lag=max_lag, min_tau=min_tau)


def phase_space_reconstruct(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Takens delay embedding 相空间重构。

    输入：
        x   : 一维时间序列
        m   : 嵌入维数
        tau : 延迟步长，单位为采样点，不是秒

    输出：
        shape = (N - (m - 1) * tau, m)

    注意：
        若窗口太短，返回空数组。
    """
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < len(x):
        # 用线性插值补 NaN，避免相空间断裂
        idx = np.arange(len(x))
        x = x.copy()
        if valid.sum() < 2:
            return np.empty((0, m))
        x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    n_vectors = len(x) - (m - 1) * tau
    if n_vectors <= 0:
        return np.empty((0, m))
    return np.column_stack([x[i * tau:i * tau + n_vectors] for i in range(m)])


def false_nearest_neighbors(
    x: np.ndarray,
    tau: int,
    max_dim: int = 10,
    rtol: float = 10.0,
    atol: float = 2.0,
    theiler: Optional[int] = None,
) -> pd.DataFrame:
    """
    False Nearest Neighbors, FNN，用于估计嵌入维数 m。

    判定逻辑：
        如果在 m 维中两个点看似相近，但升到 m+1 维后距离大幅增加，说明它们是假近邻。

    参数：
        rtol   : 距离相对增大阈值。
        atol   : 绝对距离相对于序列标准差的阈值。
        theiler: Theiler window，排除时间上过近的点，避免自相关导致假低维。

    返回：
        DataFrame，包含每个 m 的 FNN 比例。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < (max_dim + 2) * tau + 10:
        max_dim = max(2, int((len(x) - 10) // max(tau, 1)) - 1)
    if max_dim < 2:
        return pd.DataFrame(columns=["m", "FNN_Ratio", "NumVectors"])

    if theiler is None:
        theiler = tau

    sigma = np.std(x) + 1e-12
    rows = []
    for m in range(1, max_dim + 1):
        Xm = phase_space_reconstruct(x, m=m, tau=tau)
        Xm1 = phase_space_reconstruct(x, m=m + 1, tau=tau)
        n = min(len(Xm), len(Xm1))
        if n < 10:
            rows.append({"m": m, "FNN_Ratio": np.nan, "NumVectors": n})
            continue
        Xm = Xm[:n]
        Xm1 = Xm1[:n]

        tree = cKDTree(Xm)
        false_count = 0
        total = 0
        for i in range(n):
            # 查询多个近邻，因为最近的可能是自身或 Theiler window 内点
            k_query = min(n, 20)
            dists, inds = tree.query(Xm[i], k=k_query)
            if np.isscalar(inds):
                continue
            nn_idx = None
            dist_m = None
            for d, j in zip(dists, inds):
                if j == i:
                    continue
                if abs(j - i) <= theiler:
                    continue
                nn_idx = int(j)
                dist_m = float(max(d, 1e-12))
                break
            if nn_idx is None:
                continue
            dist_m1 = np.linalg.norm(Xm1[i] - Xm1[nn_idx])
            rel_increase = abs(Xm1[i, -1] - Xm1[nn_idx, -1]) / dist_m
            abs_increase = dist_m1 / sigma
            is_false = (rel_increase > rtol) or (abs_increase > atol)
            false_count += int(is_false)
            total += 1
        ratio = false_count / total if total > 0 else np.nan
        rows.append({"m": m, "FNN_Ratio": ratio, "NumVectors": total})
    return pd.DataFrame(rows)


def estimate_embedding_dimension_fnn(
    x: np.ndarray,
    tau: int,
    max_dim: int = 10,
    fnn_threshold: float = 0.05,
    rtol: float = 10.0,
    atol: float = 2.0,
    theiler: Optional[int] = None,
) -> Tuple[int, pd.DataFrame]:
    """
    根据 FNN 比例选择嵌入维数 m。

    规则：
        选择第一个 FNN_Ratio < fnn_threshold 的 m。
        如果没有达到阈值，选择 FNN_Ratio 最小的 m。
    """
    fnn_df = false_nearest_neighbors(
        x=x,
        tau=tau,
        max_dim=max_dim,
        rtol=rtol,
        atol=atol,
        theiler=theiler,
    )
    valid = fnn_df[np.isfinite(fnn_df["FNN_Ratio"])]
    if len(valid) == 0:
        return 3, fnn_df
    below = valid[valid["FNN_Ratio"] <= fnn_threshold]
    if len(below) > 0:
        return int(below.iloc[0]["m"]), fnn_df
    best = valid.sort_values("FNN_Ratio", ascending=True).iloc[0]
    return int(best["m"]), fnn_df


# ============================================================
# 3. 非线性动力学特征
# ============================================================


def sample_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """
    Sample Entropy, SampEn。

    含义：
        衡量序列复杂度和不可预测性。
        数值越大，说明局部模式越不容易重复，动力学越复杂。

    默认：
        r = 0.2 * std(x)

    注意：
        窗口太短时不稳定。建议窗口长度至少 > 200 点。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)
    if r <= 1e-12:
        return 0.0

    def _count(mm):
        templates = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        count = 0
        total = 0
        for i in range(len(templates) - 1):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += np.sum(dist <= r)
            total += len(dist)
        return count, total

    b, tb = _count(m)
    a, ta = _count(m + 1)
    if b == 0 or a == 0:
        return np.nan
    return float(-np.log((a / ta) / (b / tb) + 1e-12))


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1, normalize: bool = True) -> float:
    """
    Permutation Entropy, 排列熵。

    含义：
        衡量时间序列局部排序模式的复杂度。
        对幅值尺度不敏感，适合比较不同个体。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < order * delay + 1:
        return np.nan
    patterns = {}
    for i in range(n - delay * (order - 1)):
        window = x[i:i + delay * order:delay]
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / np.sum(counts)
    pe = -np.sum(p * np.log(p + 1e-12))
    if normalize:
        pe /= np.log(np.math.factorial(order))
    return float(pe)


def approximate_largest_lyapunov(
    X: np.ndarray,
    max_t: int = 20,
    theiler: int = 10,
    fit_start: int = 1,
    fit_end: Optional[int] = None,
) -> float:
    """
    近似最大 Lyapunov 指数。

    方法：
        使用 Rosenstein 思路：寻找相空间中非时间邻近的最近邻，观察平均距离随时间的对数增长率。

    注意：
        这是近似特征，不建议过度解释为严格物理 Lyapunov 指数。
        论文中可称为 approximate LLE 或 divergence slope。
    """
    X = np.asarray(X, dtype=float)
    if len(X) < max_t + theiler + 10:
        return np.nan
    tree = cKDTree(X)
    pairs = []
    n = len(X)
    for i in range(n - max_t):
        k_query = min(n, 30)
        dists, inds = tree.query(X[i], k=k_query)
        if np.isscalar(inds):
            continue
        for j in inds:
            j = int(j)
            if j == i:
                continue
            if abs(j - i) <= theiler:
                continue
            if j + max_t < n and i + max_t < n:
                pairs.append((i, j))
            break
    if len(pairs) < 10:
        return np.nan

    div = []
    for k in range(max_t):
        ds = []
        for i, j in pairs:
            d = np.linalg.norm(X[i + k] - X[j + k])
            if d > 1e-12 and np.isfinite(d):
                ds.append(d)
        if len(ds) < 5:
            div.append(np.nan)
        else:
            div.append(np.mean(np.log(ds)))
    div = np.asarray(div)
    valid = np.isfinite(div)
    if valid.sum() < 5:
        return np.nan
    if fit_end is None:
        fit_end = min(max_t, max(fit_start + 5, max_t // 2))
    idx = np.arange(max_t)
    fit_mask = valid & (idx >= fit_start) & (idx < fit_end)
    if fit_mask.sum() < 3:
        return np.nan
    slope = np.polyfit(idx[fit_mask], div[fit_mask], 1)[0]
    return float(slope)


def recurrence_plot_matrix(
    X: np.ndarray,
    recurrence_rate: Optional[float] = 0.05,
    epsilon: Optional[float] = None,
    metric: str = "euclidean",
    theiler: int = 1,
) -> Tuple[np.ndarray, float]:
    """
    构造 recurrence matrix。

    两种阈值方式：
        1. 指定 recurrence_rate，例如 0.05，自动选择距离分位数作为 epsilon；
        2. 指定 epsilon，固定阈值。

    推荐：
        做跨患者比较时，用固定 recurrence_rate 更稳定。
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 10:
        return np.zeros((0, 0), dtype=bool), np.nan
    D = squareform(pdist(X, metric=metric))

    # 排除 Theiler window，避免主对角线附近自相关虚高
    valid_mask = np.ones_like(D, dtype=bool)
    for k in range(-theiler, theiler + 1):
        valid_mask &= ~np.eye(n, k=k, dtype=bool)

    valid_dist = D[valid_mask]
    valid_dist = valid_dist[np.isfinite(valid_dist)]
    if len(valid_dist) == 0:
        return np.zeros_like(D, dtype=bool), np.nan

    if epsilon is None:
        if recurrence_rate is None:
            recurrence_rate = 0.05
        epsilon = float(np.quantile(valid_dist, recurrence_rate))

    R = D <= epsilon
    for k in range(-theiler, theiler + 1):
        R[np.eye(n, k=k, dtype=bool)] = False
    return R, float(epsilon)


def _line_lengths(binary_matrix: np.ndarray, direction: str = "diag", min_len: int = 2) -> List[int]:
    """
    统计 recurrence matrix 中对角线或竖直线长度。
    """
    R = np.asarray(binary_matrix, dtype=bool)
    n = R.shape[0]
    lengths = []
    if direction == "diag":
        offsets = range(-n + 1, n)
        for k in offsets:
            diag = np.diag(R, k=k)
            for s, e in get_segments(diag, min_len=min_len):
                lengths.append(e - s)
    elif direction == "vertical":
        for col in range(n):
            vec = R[:, col]
            for s, e in get_segments(vec, min_len=min_len):
                lengths.append(e - s)
    else:
        raise ValueError("direction must be 'diag' or 'vertical'.")
    return lengths


def rqa_features(
    X: np.ndarray,
    recurrence_rate: float = 0.05,
    epsilon: Optional[float] = None,
    theiler: int = 1,
    l_min: int = 2,
    v_min: int = 2,
) -> Dict[str, float]:
    """
    Recurrence Quantification Analysis, RQA 特征。

    输出：
        RR     : recurrence rate，递归率
        DET    : determinism，确定性，对角线结构比例
        LAM    : laminarity，层流性，竖直线结构比例
        L_mean : 平均对角线长度
        L_max  : 最大对角线长度
        V_mean : 平均竖直线长度
        TT     : trapping time，平均层流停留时间

    生理解释：
        - RR 高：状态重复出现较多；
        - DET 高：动力学更可预测或周期性更强；
        - LAM 高：存在停滞/平台/层流状态；
        - L_max 与系统稳定性和可预测长度相关。
    """
    R, eps = recurrence_plot_matrix(
        X,
        recurrence_rate=recurrence_rate,
        epsilon=epsilon,
        theiler=theiler,
    )
    if R.size == 0:
        return {
            "RQA_RR": np.nan,
            "RQA_DET": np.nan,
            "RQA_LAM": np.nan,
            "RQA_L_mean": np.nan,
            "RQA_L_max": np.nan,
            "RQA_V_mean": np.nan,
            "RQA_TT": np.nan,
            "RQA_epsilon": np.nan,
        }
    total_rec = R.sum()
    n = R.shape[0]
    rr = total_rec / (n * n - n + 1e-12)

    diag_lengths = _line_lengths(R, direction="diag", min_len=l_min)
    vert_lengths = _line_lengths(R, direction="vertical", min_len=v_min)

    diag_points = np.sum(diag_lengths) if len(diag_lengths) else 0
    vert_points = np.sum(vert_lengths) if len(vert_lengths) else 0

    det = diag_points / (total_rec + 1e-12)
    lam = vert_points / (total_rec + 1e-12)

    return {
        "RQA_RR": float(rr),
        "RQA_DET": float(det),
        "RQA_LAM": float(lam),
        "RQA_L_mean": float(np.mean(diag_lengths)) if diag_lengths else np.nan,
        "RQA_L_max": float(np.max(diag_lengths)) if diag_lengths else np.nan,
        "RQA_V_mean": float(np.mean(vert_lengths)) if vert_lengths else np.nan,
        "RQA_TT": float(np.mean(vert_lengths)) if vert_lengths else np.nan,
        "RQA_epsilon": float(eps),
    }


def basic_time_features(x: np.ndarray) -> Dict[str, float]:
    """
    基础时域特征。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {k: np.nan for k in [
            "Mean", "Std", "Median", "IQR", "Min", "Max", "Range",
            "Skewness", "Kurtosis", "Diff_MAD", "Diff_Energy"
        ]}
    dx = np.diff(x)
    q75, q25 = np.percentile(x, [75, 25])
    med_dx = np.median(dx)
    mad_dx = np.median(np.abs(dx - med_dx)) * 1.4826 if len(dx) else np.nan
    return {
        "Mean": float(np.mean(x)),
        "Std": float(np.std(x)),
        "Median": float(np.median(x)),
        "IQR": float(q75 - q25),
        "Min": float(np.min(x)),
        "Max": float(np.max(x)),
        "Range": float(np.max(x) - np.min(x)),
        "Skewness": float(skew(x, bias=False)) if len(x) > 3 else np.nan,
        "Kurtosis": float(kurtosis(x, fisher=False, bias=False)) if len(x) > 3 else np.nan,
        "Diff_MAD": float(mad_dx),
        "Diff_Energy": float(np.sum(dx ** 2)),
    }


def phase_space_geometry_features(X: np.ndarray) -> Dict[str, float]:
    """
    相空间几何特征。

    这些特征比严格混沌指标更稳健，适合医学小样本分类：
        - 轨迹长度
        - 平均步长
        - 状态云扩散程度
        - 协方差特征值谱
        - 轨迹体积 proxy
    """
    X = np.asarray(X, dtype=float)
    if len(X) < 3:
        return {k: np.nan for k in [
            "PS_TrajectoryLength", "PS_MeanStep", "PS_StepStd",
            "PS_StateSpread", "PS_Eig1", "PS_Eig2", "PS_EigRatio12", "PS_LogVolume"
        ]}
    steps = np.linalg.norm(np.diff(X, axis=0), axis=1)
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 1e-12))[::-1]
    log_volume = float(np.sum(np.log(eigvals)))
    eig1 = eigvals[0]
    eig2 = eigvals[1] if len(eigvals) > 1 else np.nan
    return {
        "PS_TrajectoryLength": float(np.sum(steps)),
        "PS_MeanStep": float(np.mean(steps)),
        "PS_StepStd": float(np.std(steps)),
        "PS_StateSpread": float(np.mean(np.linalg.norm(X - np.mean(X, axis=0), axis=1))),
        "PS_Eig1": float(eig1),
        "PS_Eig2": float(eig2) if np.isfinite(eig2) else np.nan,
        "PS_EigRatio12": float(eig1 / (eig2 + 1e-12)) if np.isfinite(eig2) else np.nan,
        "PS_LogVolume": log_volume,
    }


# ============================================================
# 4. 单窗口特征提取与批量特征表
# ============================================================


def estimate_embedding_parameters(
    x: np.ndarray,
    tau_method: str = "ami",
    max_tau: int = 120,
    max_dim: int = 8,
    fnn_threshold: float = 0.05,
    default_tau: int = 1,
    default_m: int = 3,
) -> Tuple[int, int, pd.DataFrame]:
    """
    统一估计相空间重构参数 tau 和 m。

    tau_method:
        'ami'      : Average Mutual Information 第一个局部极小值；
        'autocorr' : 自相关 1/e。

    返回：
        tau, m, fnn_df
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 100 or np.std(x) < 1e-12:
        return default_tau, default_m, pd.DataFrame()

    max_tau = int(min(max_tau, max(2, len(x) // 10)))
    if tau_method == "ami":
        tau = estimate_tau_ami(x, max_lag=max_tau)
    elif tau_method == "autocorr":
        tau = estimate_tau_autocorr(x, max_lag=max_tau)
    else:
        raise ValueError("tau_method must be 'ami' or 'autocorr'.")

    tau = int(max(1, tau))
    m, fnn_df = estimate_embedding_dimension_fnn(
        x,
        tau=tau,
        max_dim=max_dim,
        fnn_threshold=fnn_threshold,
        theiler=tau,
    )
    if not np.isfinite(m) or m < 1:
        m = default_m
    return int(tau), int(m), fnn_df


def extract_phase_space_features_for_signal(
    x: np.ndarray,
    signal_name: str,
    tau: Optional[int] = None,
    m: Optional[int] = None,
    tau_method: str = "ami",
    max_tau: int = 120,
    max_dim: int = 8,
    normalize: str = "robust_zscore",
    recurrence_rate: float = 0.05,
) -> Dict[str, float]:
    """
    对单个一维信号窗口提取相空间和非线性特征。

    signal_name:
        用作特征名前缀，例如 'Clean', 'VMDRecon', 'Mode3'。

    normalize:
        'robust_zscore' 推荐默认；
        'none' 不标准化；
        'minmax' 用于可视化，不推荐分类默认。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    prefix = signal_name

    # 长度不足时返回 NaN 特征，保证批处理不中断。
    if len(x) < 100 or np.std(x) < 1e-12:
        out = {f"{prefix}_NumPoints": len(x)}
        out.update({f"{prefix}_{k}": np.nan for k in [
            "Tau", "EmbeddingDim", "SampEn", "PermEn", "ApproxLLE",
            "RQA_RR", "RQA_DET", "RQA_LAM", "PS_TrajectoryLength"
        ]})
        return out

    if normalize == "robust_zscore":
        xn = robust_zscore(x)
    elif normalize == "minmax":
        xn = minmax_scale(x)
    elif normalize == "none":
        xn = x.copy()
    else:
        raise ValueError("normalize must be 'robust_zscore', 'minmax', or 'none'.")

    if tau is None or m is None:
        est_tau, est_m, _ = estimate_embedding_parameters(
            xn,
            tau_method=tau_method,
            max_tau=max_tau,
            max_dim=max_dim,
        )
        tau = est_tau if tau is None else tau
        m = est_m if m is None else m

    X = phase_space_reconstruct(xn, m=int(m), tau=int(tau))

    out = {
        f"{prefix}_NumPoints": int(len(x)),
        f"{prefix}_Tau": int(tau),
        f"{prefix}_EmbeddingDim": int(m),
        f"{prefix}_EmbeddedVectors": int(len(X)),
    }

    # 基础时域特征用未标准化原信号，保留幅值意义。
    for k, v in basic_time_features(x).items():
        out[f"{prefix}_{k}"] = v

    # 熵和相空间特征用标准化信号，更适合跨患者比较。
    out[f"{prefix}_SampEn"] = sample_entropy(xn, m=2, r=0.2 * np.std(xn))
    out[f"{prefix}_PermEn"] = permutation_entropy(xn, order=3, delay=max(1, int(tau)))

    if len(X) >= 30:
        for k, v in phase_space_geometry_features(X).items():
            out[f"{prefix}_{k}"] = v
        for k, v in rqa_features(X, recurrence_rate=recurrence_rate, theiler=max(1, int(tau))).items():
            out[f"{prefix}_{k}"] = v
        out[f"{prefix}_ApproxLLE"] = approximate_largest_lyapunov(
            X,
            max_t=min(30, max(10, len(X) // 10)),
            theiler=max(1, int(tau)),
        )
    else:
        # 嵌入向量太少时，RQA和LLE不可靠。
        for name in [
            "PS_TrajectoryLength", "PS_MeanStep", "PS_StepStd", "PS_StateSpread",
            "PS_Eig1", "PS_Eig2", "PS_EigRatio12", "PS_LogVolume",
            "RQA_RR", "RQA_DET", "RQA_LAM", "RQA_L_mean", "RQA_L_max",
            "RQA_V_mean", "RQA_TT", "RQA_epsilon", "ApproxLLE",
        ]:
            out[f"{prefix}_{name}"] = np.nan

    return out


def build_phase_space_feature_table(
    df: pd.DataFrame,
    windows: List[WindowSpec],
    signal_columns: Sequence[str],
    time_col: str = "Time",
    subject_id: Optional[str] = None,
    label: Optional[Union[int, str]] = None,
    tau_method: str = "ami",
    max_tau: int = 120,
    max_dim: int = 8,
    normalize: str = "robust_zscore",
    recurrence_rate: float = 0.05,
    global_tau_m: Optional[Dict[str, Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """
    对一组窗口和多个信号列批量提取相空间特征。

    signal_columns:
        例如：
            ['Pressure_Clean', 'VMD_Reconstructed', 'VMD_Mode3', 'VMD_Mode4']

    global_tau_m:
        可选。如果你希望全队列使用同一套 tau/m，传入：
            {'Pressure_Clean': (tau, m), 'VMD_Mode3': (tau, m)}
        这样跨患者特征更可比。

    SCI建议：
        - 探索阶段可每个窗口自适应 tau/m；
        - 最终分类论文建议固定 tau/m 或在训练集估计后应用到测试集，避免信息泄露。
    """
    rows = []
    t = df[time_col].to_numpy(dtype=float)

    for w in windows:
        row = {
            "SubjectID": subject_id,
            "Label": label,
            "WindowID": w.window_id,
            "WindowMethod": w.method,
            "StartTime": w.start_time,
            "EndTime": w.end_time,
            "CenterTime": w.center_time,
            "StartIndex": w.start_idx,
            "EndIndex": w.end_idx,
            "NumPoints": w.num_points,
            "EventID": w.event_id,
            "EventStartTime": w.event_start_time,
            "EventEndTime": w.event_end_time,
            "EventCoverage": w.event_coverage,
        }

        for col in signal_columns:
            if col not in df.columns:
                continue
            x = df[col].to_numpy(dtype=float)[w.start_idx:w.end_idx]
            tau = None
            m = None
            if global_tau_m is not None and col in global_tau_m:
                tau, m = global_tau_m[col]
            feats = extract_phase_space_features_for_signal(
                x,
                signal_name=col,
                tau=tau,
                m=m,
                tau_method=tau_method,
                max_tau=max_tau,
                max_dim=max_dim,
                normalize=normalize,
                recurrence_rate=recurrence_rate,
            )
            row.update(feats)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 5. VMD结果接入辅助函数
# ============================================================


def add_vmd_modes_to_dataframe(
    df: pd.DataFrame,
    vmd_result,
    mode_numbers: Sequence[int],
    reconstructed_col: str = "VMD_Reconstructed",
    mode_prefix: str = "VMD_Mode",
) -> pd.DataFrame:
    """
    将 VMD 重构信号和指定 mode 加入 df，便于后续窗口特征提取。

    参数：
        vmd_result:
            physiology_constrained_vmd_reconstruction 的输出。
        mode_numbers:
            例如 [1,2,3,4]。
    """
    out = df.copy()
    n = min(len(out), len(vmd_result.reconstructed))

    out = out.iloc[:n].copy()
    rec = np.asarray(vmd_result.reconstructed, dtype=float)[:n]
    # 保证重构信号均值与 Pressure_Clean 对齐，避免基线偏移。
    if "Pressure_Clean" in out.columns:
        ref = out["Pressure_Clean"].to_numpy(dtype=float)[:n]
        rec = rec - np.nanmean(rec) + np.nanmean(ref)
    out[reconstructed_col] = rec

    modes = np.asarray(vmd_result.modes, dtype=float)
    for mode_number in mode_numbers:
        idx = mode_number - 1
        if idx < 0 or idx >= modes.shape[0]:
            continue
        mode = modes[idx, :n]
        out[f"{mode_prefix}{mode_number}"] = mode
    return out


def recommend_phase_space_signals(
    include_clean: bool = True,
    include_vmd_reconstructed: bool = True,
    include_modes: Sequence[int] = (3, 4),
) -> List[str]:
    """
    推荐用于相空间特征提取的信号列。

    默认重点分析：
        - Pressure_Clean：整体清洗后动力学；
        - VMD_Reconstructed：生理相关 modes 重构；
        - VMD_Mode3：核心事件相关模态；
        - VMD_Mode4：高频复杂动力学模态。
    """
    cols = []
    if include_clean:
        cols.append("Pressure_Clean")
    if include_vmd_reconstructed:
        cols.append("VMD_Reconstructed")
    for m in include_modes:
        cols.append(f"VMD_Mode{m}")
    return cols


# ============================================================
# 6. 质量控制与特征筛选辅助
# ============================================================


def summarize_feature_missingness(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总特征缺失率，便于删除不稳定特征。
    """
    rows = []
    for col in feature_df.columns:
        miss = feature_df[col].isna().mean()
        rows.append({"Feature": col, "MissingRate": miss})
    return pd.DataFrame(rows).sort_values("MissingRate", ascending=False)


def drop_high_missing_features(
    feature_df: pd.DataFrame,
    max_missing_rate: float = 0.3,
    protected_cols: Sequence[str] = (
        "SubjectID", "Label", "WindowID", "WindowMethod",
        "StartTime", "EndTime", "CenterTime", "EventID"
    ),
) -> pd.DataFrame:
    """
    删除缺失率过高的特征列。
    """
    out = feature_df.copy()
    drop_cols = []
    for col in out.columns:
        if col in protected_cols:
            continue
        if out[col].isna().mean() > max_missing_rate:
            drop_cols.append(col)
    return out.drop(columns=drop_cols)
