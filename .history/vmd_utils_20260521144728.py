#数组计算库
import numpy as np
#处理表格
import pandas as pd
#导入savgol_filter平滑滤波器，
from scipy.signal import savgol_filter


# ============================================================
# 工具函数（预处理）
# ============================================================

def get_continuous_segments(mask):
    """
    输入 bool mask，输出连续 True 区间。
    返回格式：[{"start_pos": s, "end_pos": e}, ...]
    其中 end_pos 是包含端点。
    """
    mask = np.asarray(mask, dtype=bool)
    segments = []

    n = len(mask)
    i = 0

    while i < n:
        if not mask[i]:
            i += 1
            continue

        start = i

        while i < n and mask[i]:
            i += 1

        end = i - 1

        segments.append({
            "start_pos": start,
            "end_pos": end
        })

    return segments

#硬阈值
def repair_pressure_by_hard_range(
    data,
    time_col="Time",
    pressure_col="Pressure",
    pressure_min=85,
    pressure_max=130,
    output_col="Pressure_RangeRepaired"
):
    df = data.copy()

    t = df[time_col].to_numpy(dtype=float)
    p = df[pressure_col].to_numpy(dtype=float)

    abnormal_mask = (
        (p < pressure_min) |
        (p > pressure_max) |
        (~np.isfinite(p)) |
        (~np.isfinite(t))
    )

    valid_mask = (~abnormal_mask) & np.isfinite(t) & np.isfinite(p)

    p_repaired = p.copy()

    if valid_mask.sum() < 2:
        raise ValueError("有效正常压力点少于 2 个，无法插值修复。")

    p_repaired[abnormal_mask] = np.interp(
        t[abnormal_mask],
        t[valid_mask],
        p[valid_mask]
    )

    df[output_col] = p_repaired
    df["RangeAbnormalMask"] = abnormal_mask

    return df

#把窗口长度变成奇数
def _odd(n: int, minimum: int = 3) -> int:
    n = int(max(n, minimum))
    return n if n % 2 == 1 else n + 1

#估计采样间隔
def _estimate_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)#把输入转换成numpy数组，并保证是浮点数
    dt = np.diff(t)#计算相邻时间差
    dt = dt[np.isfinite(dt) & (dt > 0)]#只保留有效的，正的时间间隔

    if len(dt) == 0:
        raise ValueError("Time column must contain at least two increasing values.")

    return float(np.median(dt))

#把秒数转换成采样点数 seconds窗口长度，单位秒
def _seconds_to_samples(seconds: float, dt: float, minimum: int = 3, odd: bool = True) -> int:
    n = int(round(seconds / dt))
    #如果要求用奇数，就调用_odd()
    if odd:
        return _odd(n, minimum)

    return int(max(n, minimum))

#计算局部中位数
#把 numpy 数组转成 pandas Series，因为 pandas 的 rolling 操作很方便
#min_periods保证至少 window // 3 个有效点才计算，避免边缘失真
def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(x)
        .rolling(window=window, center=True, min_periods=max(3, window // 3))
        .median()
        .bfill()
        .ffill()
        .to_numpy()
    )

#计算局部MAD（中位数绝对偏差）噪声尺度
def _rolling_mad(x: np.ndarray, window: int, scale: bool = True, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=float)#转换成浮点数
    #计算一个窗口内的MAD
    def mad_func(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]#去Nan和inf

        if len(v) == 0:
            return np.nan

        med = np.median(v)
        return np.median(np.abs(v - med))#计算窗口内每个值偏离中位数的绝对值，然后再取中位数。=MAD

    mad = (
        pd.Series(x)
        .rolling(window=window, center=True, min_periods=max(3, window // 3))
        .apply(mad_func, raw=True)
        .bfill()
        .ffill()
        .to_numpy()
    )
    #是否把MAD转换成近似标准差
    if scale:
        mad = 1.4826 * mad

    finite_x = x[np.isfinite(x)]

    if len(finite_x) > 0:
        global_mad = np.median(np.abs(finite_x - np.median(finite_x)))
        if scale:
            global_mad *= 1.4826
    else:
        global_mad = eps#如果没有数据就用极小值
    #设置局部噪声下限，如果某一小段非常平稳，局部MAD可能接近0，此时阈值也会接近0，导致一点点正常波动都被误判成异常
    # #所以设置局部噪声不能小于全局噪声的5% 
    floor = max(eps, 0.05 * global_mad)
    mad = np.nan_to_num(mad, nan=floor, posinf=floor, neginf=floor)#如果MAD里有NaN或inf，用floor替换

    return np.maximum(mad, floor)#每个位置都不低于下限

#返回所有连续True区间
def _components(mask: np.ndarray):
    """
    返回布尔 mask 中连续 True 区间。
    每个区间为 [start, end)，end 不包含。
    """
    mask = np.asarray(mask, dtype=bool)
    out = []
    i, n = 0, len(mask)

    while i < n:
        if not mask[i]:
            i += 1
            continue

        s = i

        while i < n and mask[i]:
            i += 1

        out.append((s, i))

    return out

#把两个相邻事件（True）中间的（False）区间补上（True）
def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()

    if max_gap <= 0:
        return mask

    comps = _components(mask)

    for (s1, e1), (s2, e2) in zip(comps[:-1], comps[1:]):
        if s2 - e1 <= max_gap:
            mask[e1:s2] = True

    return mask


def _expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)

    if radius <= 0 or mask.sum() == 0:
        return mask.copy()

    out = mask.copy()
    idx = np.where(mask)[0]
    n = len(mask)

    for i in idx:
        out[max(0, i - radius):min(n, i + radius + 1)] = True

    return out

#把原时间轴上的mask投影到重采样后的同一时间轴上
def _project_mask_to_uniform(
    t_raw: np.ndarray,
    mask_raw: np.ndarray,
    t_uniform: np.ndarray,
    pad_s: float = 0.0
) -> np.ndarray:
    out = np.zeros(len(t_uniform), dtype=bool)

    for s, e in _components(mask_raw):
        left = t_raw[s] - pad_s
        right = t_raw[e - 1] + pad_s
        out |= (t_uniform >= left) & (t_uniform <= right)

    return out

#只对mask=true的异常区域做插值修复
def _interpolate_over_mask(
    t: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    max_gap_s: float = np.inf
):
    """
    只重建 mask=True 的区域。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    y_out = y.copy()
    low_conf = np.zeros(len(y), dtype=bool)

    y_temp = y.copy()
    y_temp[mask] = np.nan

    for s, e in _components(mask):
        dt = _estimate_dt(t)
        duration = t[e - 1] - t[s] + dt if e > s else 0.0

        valid = np.isfinite(y_temp)

        if valid.sum() < 2:
            y_out[s:e] = np.nan
            low_conf[s:e] = True
            continue

        y_out[s:e] = np.interp(t[s:e], t[valid], y_temp[valid])

        has_left = valid[:s].any()
        has_right = valid[e:].any()

        if duration > max_gap_s or (not has_left) or (not has_right):
            low_conf[s:e] = True

    return y_out, low_conf

#整理原始输入data
def _prepare_dataframe(data, time_col, pressure_col, ph_col=None, temp_col=None):
    cols = [time_col, pressure_col]

    if ph_col is not None and ph_col in data.columns:
        cols.append(ph_col)

    if temp_col is not None and temp_col in data.columns:
        cols.append(temp_col)

    df = data[cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[time_col, pressure_col])
    df = df.sort_values(time_col)

    # 重复时间点取平均，保证时间轴严格递增
    df = df.groupby(time_col, as_index=False).mean(numeric_only=True)

    t = df[time_col].to_numpy(dtype=float)
    p = df[pressure_col].to_numpy(dtype=float)

    if len(t) < 7:
        raise ValueError("有效压力点太少，无法进行自适应去噪。")

    if not np.all(np.diff(t) > 0):
        raise ValueError("Time 去重和排序后仍然不是严格递增。")

    return df, t, p


# ============================================================
# 新增 Step 0：硬阈值物理范围门控
# ============================================================

def _apply_hard_pressure_range_gate(
    t,
    p,
    pressure_min=85.0,
    pressure_max=130.0,
    max_gap_s=30.0
):
    """
    在任何滤波、重采样、平滑之前，先处理明确超出物理范围的压力点。

    你的情况：
    Pressure < 85 或 Pressure > 130 的点直接判为异常伪影。

    返回：
    p_repaired          : 修复后的压力
    range_abnormal_mask : 原始时间轴上的硬阈值异常 mask
    low_conf_mask       : 低置信度修复 mask
    """

    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)

    abnormal = (~np.isfinite(p)) | (~np.isfinite(t))

    if pressure_min is not None:
        abnormal |= p < pressure_min

    if pressure_max is not None:
        abnormal |= p > pressure_max

    p_repaired = p.copy()
    low_conf = np.zeros(len(p), dtype=bool)

    valid = (~abnormal) & np.isfinite(t) & np.isfinite(p)

    if abnormal.sum() == 0:
        return p_repaired, abnormal, low_conf

    if valid.sum() < 2:
        raise ValueError("硬阈值门控后有效正常压力点少于 2 个，无法插值修复。")

    dt = _estimate_dt(t)

    for s, e in _components(abnormal):
        duration = t[e - 1] - t[s] + dt

        p_repaired[s:e] = np.interp(
            t[s:e],
            t[valid],
            p[valid]
        )

        has_left = valid[:s].any()
        has_right = valid[e:].any()

        if duration > max_gap_s or (not has_left) or (not has_right):
            low_conf[s:e] = True

    return p_repaired, abnormal, low_conf


# ============================================================
# Step 2：局部噪声估计
# ============================================================

def _estimate_local_noise(t, p, noise_window_s):
    """
    用一阶差分估计局部噪声。

    sigma_p   : 压力点噪声水平
    sigma_dp  : 相邻点跳变噪声水平
    sigma_ddp : 曲率/二阶差分噪声水平
    """
    dt = _estimate_dt(t)
    w = _seconds_to_samples(noise_window_s, dt, minimum=7, odd=True)

    p_filled = np.asarray(p, dtype=float).copy()
    valid = np.isfinite(p_filled)

    if valid.sum() >= 2 and not valid.all():
        p_filled[~valid] = np.interp(t[~valid], t[valid], p_filled[valid])

    dp = np.diff(p_filled, prepend=p_filled[0])
    ddp = np.diff(dp, prepend=dp[0])

    sigma_dp = _rolling_mad(dp, w, scale=True)
    sigma_ddp = _rolling_mad(ddp, w, scale=True)

    sigma_p = sigma_dp / np.sqrt(2.0)

    return sigma_p, sigma_dp, sigma_ddp


# ============================================================
# Step 3：生理事件保护层
# ============================================================

def _detect_physio_events(
    t,
    p,
    baseline,
    sigma_p,
    min_event_duration_s,
    event_amp_k,
    event_area_k,
    max_event_gap_s
):
    dt = _estimate_dt(t)
    residual = p - baseline

    candidate = residual > event_amp_k * sigma_p

    gap_n = _seconds_to_samples(max_event_gap_s, dt, minimum=1, odd=False)
    candidate = _fill_short_gaps(candidate, max_gap=gap_n)

    event_mask = np.zeros(len(p), dtype=bool)
    event_score = np.zeros(len(p), dtype=float)

    min_event_n = _seconds_to_samples(min_event_duration_s, dt, minimum=2, odd=False)

    for s, e in _components(candidate):
        dur_n = e - s
        dur_s = t[e - 1] - t[s] + dt

        if dur_n < min_event_n:
            continue

        seg_res = residual[s:e]
        seg_sigma = np.median(sigma_p[s:e])

        peak = np.max(seg_res)

        if dur_n > 1:
            area = np.trapz(np.maximum(seg_res, 0), t[s:e])
        else:
            area = peak * dt

        min_area = event_area_k * seg_sigma * max(dur_s, dt)

        if peak > event_amp_k * seg_sigma and area > min_area:
            event_mask[s:e] = True
            event_score[s:e] = np.maximum(
                seg_res / (event_amp_k * sigma_p[s:e] + 1e-9),
                0
            )

    return event_mask, event_score


# ============================================================
# Step 4：形态学伪影检测
# ============================================================

def _detect_artifacts_raw(
    t,
    p,
    baseline,
    sigma_p,
    sigma_dp,
    sigma_ddp,
    physio_event_mask,
    max_artifact_duration_s,
    amp_k,
    slope_k,
    curvature_k,
    min_abs_spike,
    neighbor_return_k,
    flatline_min_duration_s,
    flatline_eps,
):
    n = len(p)
    dt = _estimate_dt(t)
    residual = p - baseline

    artifact = np.zeros(n, dtype=bool)
    reason = np.array(["ok"] * n, dtype=object)

    amp_thr = np.maximum(amp_k * sigma_p, min_abs_spike)
    slope_thr = np.maximum(slope_k * sigma_dp, min_abs_spike)
    curv_thr = np.maximum(curvature_k * sigma_ddp, min_abs_spike)

    # ------------------------------------------------------------
    # 1. 单点孤立尖刺
    # ------------------------------------------------------------
    for i in range(1, n - 1):
        expected = np.interp(
            t[i],
            [t[i - 1], t[i + 1]],
            [p[i - 1], p[i + 1]]
        )

        dev = abs(p[i] - expected)

        jump_left = p[i] - p[i - 1]
        jump_right = p[i + 1] - p[i]

        reverse = jump_left * jump_right < 0

        local_sig = np.median(sigma_p[max(0, i - 2):min(n, i + 3)])

        neighbors_close = (
            abs(p[i - 1] - p[i + 1])
            <= neighbor_return_k * local_sig + min_abs_spike
        )

        if (
            reverse
            and dev > amp_thr[i]
            and abs(jump_left) > slope_thr[i]
            and abs(jump_right) > slope_thr[i]
            and neighbors_close
        ):
            artifact[i] = True
            reason[i] = "isolated_spike"

    # ------------------------------------------------------------
    # 2. 曲率异常
    # ------------------------------------------------------------
    dp = np.diff(p, prepend=p[0])
    ddp = np.diff(dp, prepend=dp[0])

    curvature_candidate = np.abs(ddp) > curv_thr

    for i in np.where(curvature_candidate)[0]:
        if 1 <= i < n - 1:
            expected = np.interp(
                t[i],
                [t[i - 1], t[i + 1]],
                [p[i - 1], p[i + 1]]
            )

            if abs(p[i] - expected) > amp_thr[i]:
                artifact[i] = True
                reason[i] = "curvature_spike"

    # ------------------------------------------------------------
    # 3. 短时残差异常段
    # ------------------------------------------------------------
    short_outlier = np.abs(residual) > amp_thr

    max_artifact_n = _seconds_to_samples(
        max_artifact_duration_s,
        dt,
        minimum=1,
        odd=False
    )

    for s, e in _components(short_outlier):
        dur_n = e - s
        dur_s = t[e - 1] - t[s] + dt

        if dur_n > max_artifact_n or dur_s > max_artifact_duration_s + dt:
            continue

        overlap = physio_event_mask[s:e].mean() if e > s else 0.0

        if overlap > 0.5 and dur_s > max_artifact_duration_s:
            continue

        left = s - 1
        right = e

        if left < 0 or right >= n:
            continue

        boundary_gap = abs(p[left] - p[right])
        local_sig = np.median(sigma_p[s:e])
        peak_dev = np.max(np.abs(residual[s:e]))

        returns_to_baseline = (
            boundary_gap <= neighbor_return_k * local_sig + min_abs_spike
        )

        needle_like = peak_dev > np.median(amp_thr[s:e])

        if returns_to_baseline and needle_like:
            artifact[s:e] = True
            reason[s:e] = "short_pulse"

    # ------------------------------------------------------------
    # 4. 短时反向大边缘
    # ------------------------------------------------------------
    d = np.diff(p)

    edge_thr = np.maximum(slope_thr[:-1], slope_thr[1:])
    large_edge = np.abs(d) > edge_thr
    edge_idx = np.where(large_edge)[0]

    for i in edge_idx:
        j_max = min(n - 2, i + max_artifact_n + 1)

        for j in range(i + 1, j_max + 1):
            if not large_edge[j]:
                continue

            if d[i] * d[j] >= 0:
                continue

            s = i + 1
            e = j + 1

            if e <= s:
                continue

            bridge = np.interp(
                t[s:e],
                [t[i], t[j + 1]],
                [p[i], p[j + 1]]
            )

            protrusion = np.max(np.abs(p[s:e] - bridge))
            local_thr = np.median(amp_thr[s:e])

            if protrusion > local_thr:
                artifact[s:e] = True
                reason[s:e] = "reverse_edge_pulse"

            break

    # ------------------------------------------------------------
    # 5. 丢包样平台段
    # ------------------------------------------------------------
    if flatline_min_duration_s is not None and flatline_min_duration_s > 0:
        flat = np.abs(np.diff(p, prepend=p[0])) <= flatline_eps

        min_flat_n = _seconds_to_samples(
            flatline_min_duration_s,
            dt,
            minimum=2,
            odd=False
        )

        for s, e in _components(flat):
            if e - s < min_flat_n:
                continue

            left_jump = abs(p[s] - p[s - 1]) if s > 0 else 0.0
            right_jump = abs(p[e] - p[e - 1]) if e < n else 0.0
            local_sig = np.median(sigma_p[s:e])

            if left_jump > amp_k * local_sig or right_jump > amp_k * local_sig:
                artifact[s:e] = True
                reason[s:e] = "flatline_dropout"

    # ------------------------------------------------------------
    # 最终保护：不把长时间生理事件整体删掉
    # ------------------------------------------------------------
    refined = np.zeros(n, dtype=bool)
    refined_reason = np.array(["ok"] * n, dtype=object)

    for s, e in _components(artifact):
        dur_s = t[e - 1] - t[s] + dt
        overlap = physio_event_mask[s:e].mean() if e > s else 0.0

        if dur_s <= max_artifact_duration_s + dt:
            refined[s:e] = True
            refined_reason[s:e] = reason[s:e]

        elif overlap < 0.2:
            refined[s:e] = True
            refined_reason[s:e] = reason[s:e]

    return refined, refined_reason


# ============================================================
# Step 7：事件保护型自适应平滑
# ============================================================

def _event_preserving_smooth(
    t,
    p,
    physio_event_mask,
    event_score,
    smooth_window_s
):
    p = np.asarray(p, dtype=float)
    n = len(p)
    dt = _estimate_dt(t)

    valid = np.isfinite(p)

    if valid.sum() < 2:
        return p.copy()

    p_filled = p.copy()
    p_filled[~valid] = np.interp(t[~valid], t[valid], p[valid])

    smooth_w = _seconds_to_samples(
        smooth_window_s,
        dt,
        minimum=5,
        odd=True
    )

    if smooth_w >= n:
        smooth_w = n - 1 if (n - 1) % 2 == 1 else n - 2

    if smooth_w >= 5:
        p_smooth = savgol_filter(
            p_filled,
            window_length=smooth_w,
            polyorder=2,
            mode="interp"
        )
    else:
        p_smooth = p_filled.copy()

    w = np.clip(event_score, 0, 1)

    # 生理事件核心区强保护
    w[physio_event_mask] = np.maximum(w[physio_event_mask], 0.85)

    # 生理事件边缘保护
    edge_mask = _expand_mask(physio_event_mask, radius=1)
    w[edge_mask] = np.maximum(w[edge_mask], 0.65)

    p_final = w * p_filled + (1 - w) * p_smooth
    p_final[~valid] = np.nan

    return p_final


# ============================================================
# 主函数：加入硬阈值门控后的完整版本
# ============================================================

def physiology_preserving_adaptive_denoise(
    data,
    time_col="Time",
    pressure_col="Pressure",
    ph_col="PH",
    temp_col="Temperature",

    # resample_dt=None 表示使用原始中位采样间隔
    resample_dt=None,

    # 新增：硬阈值压力范围门控
    use_hard_range_gate=True,
    pressure_min=95.0,
    pressure_max=120.0,
    hard_range_max_gap_s=30.0,

    # baseline 与局部噪声窗口
    baseline_window_s=60.0,
    noise_window_s=20.0,

    # 生理事件保护参数
    min_event_duration_s=6.0,
    event_amp_k=3.0,
    event_area_k=1.5,
    max_event_gap_s=2.4,

    # 伪影检测参数
    max_artifact_duration_s=4.8,
    amp_k=6.0,
    slope_k=6.0,
    curvature_k=6.0,
    min_abs_spike=3.0,
    neighbor_return_k=4.0,

    # 丢包/平台段
    flatline_min_duration_s=12.0,
    flatline_eps=1e-6,

    # 重建和平滑
    max_reconstruct_gap_s=8.0,
    smooth_window_s=4.8,

    # 迭代次数
    n_iter=2
):
    """
    Physiology-preserving adaptive denoising framework.

    改动重点：
    1. 先在原始时间轴上执行硬阈值门控：
       Pressure < pressure_min 或 Pressure > pressure_max 直接判为异常。
    2. 对硬阈值异常点先插值修复。
    3. 再进入 physiology-preserving adaptive denoising。
    4. 输出 RangeAbnormalMask，方便确认 95–120 外的点是否已被识别。
    """

    # ============================================================
    # Step 1. 原始数据整理
    # ============================================================
    df, t_raw, p_original = _prepare_dataframe(
        data,
        time_col=time_col,
        pressure_col=pressure_col,
        ph_col=ph_col,
        temp_col=temp_col
    )

    dt_raw = _estimate_dt(t_raw)

    if resample_dt is None:
        resample_dt = dt_raw

    # ============================================================
    # Step 1.5 新增：硬阈值物理范围门控
    # ============================================================
    if use_hard_range_gate:
        p_range_fixed, range_abnormal_raw, range_low_conf_raw = _apply_hard_pressure_range_gate(
            t_raw,
            p_original,
            pressure_min=pressure_min,
            pressure_max=pressure_max,
            max_gap_s=hard_range_max_gap_s
        )
    else:
        p_range_fixed = p_original.copy()
        range_abnormal_raw = np.zeros(len(p_original), dtype=bool)
        range_low_conf_raw = np.zeros(len(p_original), dtype=bool)

    # 后续所有自适应去噪都基于硬阈值修复后的压力
    p_work = p_range_fixed.copy()

    total_artifact_raw = range_abnormal_raw.copy()
    artifact_reason_raw = np.array(["ok"] * len(p_work), dtype=object)
    artifact_reason_raw[range_abnormal_raw] = "hard_range_outlier"

    low_conf_raw = range_low_conf_raw.copy()

    # ============================================================
    # Step 2-5. 迭代执行：
    # baseline/noise -> physio mask -> artifact mask -> reconstruction
    # ============================================================
    for _ in range(max(1, int(n_iter))):

        baseline_w = _seconds_to_samples(
            baseline_window_s,
            dt_raw,
            minimum=9,
            odd=True
        )

        baseline = _rolling_median(p_work, baseline_w)

        sigma_p, sigma_dp, sigma_ddp = _estimate_local_noise(
            t_raw,
            p_work,
            noise_window_s=noise_window_s
        )

        physio_mask_raw, event_score_raw = _detect_physio_events(
            t_raw,
            p_work,
            baseline,
            sigma_p,
            min_event_duration_s=min_event_duration_s,
            event_amp_k=event_amp_k,
            event_area_k=event_area_k,
            max_event_gap_s=max_event_gap_s
        )

        artifact_raw, reason_raw = _detect_artifacts_raw(
            t_raw,
            p_work,
            baseline,
            sigma_p,
            sigma_dp,
            sigma_ddp,
            physio_mask_raw,
            max_artifact_duration_s=max_artifact_duration_s,
            amp_k=amp_k,
            slope_k=slope_k,
            curvature_k=curvature_k,
            min_abs_spike=min_abs_spike,
            neighbor_return_k=neighbor_return_k,
            flatline_min_duration_s=flatline_min_duration_s,
            flatline_eps=flatline_eps
        )
# ============================================================
# Event-protected ArtifactMask expansion
# 只在非生理事件区域扩展伪影，避免破坏事件边界
# ============================================================

        artifact_expanded = _expand_mask(
           artifact_raw,
            radius=1
        )

        artifact_raw = artifact_expanded & (~physio_mask_raw)
# =====================================================================
        new_artifacts = artifact_raw & (~total_artifact_raw)

        if new_artifacts.sum() == 0:
            break

        p_work, low_conf_iter = _interpolate_over_mask(
            t_raw,
            p_work,
            artifact_raw,
            max_gap_s=max_reconstruct_gap_s
        )

        total_artifact_raw |= artifact_raw
        low_conf_raw |= low_conf_iter

        update_idx = artifact_raw & (artifact_reason_raw == "ok")
        artifact_reason_raw[update_idx] = reason_raw[update_idx]

    # ============================================================
    # 最终重新估计 baseline、noise、physio mask
    # ============================================================
    baseline_w = _seconds_to_samples(
        baseline_window_s,
        dt_raw,
        minimum=9,
        odd=True
    )

    baseline_raw = _rolling_median(p_work, baseline_w)

    sigma_p_raw, _, _ = _estimate_local_noise(
        t_raw,
        p_work,
        noise_window_s
    )

    physio_mask_raw, event_score_raw = _detect_physio_events(
        t_raw,
        p_work,
        baseline_raw,
        sigma_p_raw,
        min_event_duration_s=min_event_duration_s,
        event_amp_k=event_amp_k,
        event_area_k=event_area_k,
        max_event_gap_s=max_event_gap_s
    )

    # ============================================================
    # Step 6. 修复伪影后再重采样
    # ============================================================
    t_uniform = np.arange(
        t_raw[0],
        t_raw[-1] + 0.5 * resample_dt,
        resample_dt
    )

    t_uniform = t_uniform[t_uniform <= t_raw[-1]]

    valid_work = np.isfinite(p_work)

    if valid_work.sum() < 2:
        raise ValueError("伪影处理后有效点太少，无法插值。")

    pressure_raw_uniform = np.interp(t_uniform, t_raw, p_original)

    pressure_range_fixed_uniform = np.interp(
        t_uniform,
        t_raw,
        p_range_fixed
    )

    pressure_preclean_uniform = np.interp(
        t_uniform,
        t_raw[valid_work],
        p_work[valid_work]
    )

    baseline_uniform = np.interp(t_uniform, t_raw, baseline_raw)
    sigma_uniform = np.interp(t_uniform, t_raw, sigma_p_raw)

    range_abnormal_uniform = _project_mask_to_uniform(
        t_raw,
        range_abnormal_raw,
        t_uniform,
        pad_s=0.5 * resample_dt
    )

    artifact_uniform = _project_mask_to_uniform(
        t_raw,
        total_artifact_raw,
        t_uniform,
        pad_s=0.5 * resample_dt
    )

    low_conf_uniform = _project_mask_to_uniform(
        t_raw,
        low_conf_raw,
        t_uniform,
        pad_s=0.5 * resample_dt
    )

    physio_uniform = _project_mask_to_uniform(
        t_raw,
        physio_mask_raw,
        t_uniform,
        pad_s=0.5 * resample_dt
    )

    event_score_uniform = np.interp(t_uniform, t_raw, event_score_raw)

    # PH 和温度只同步到统一时间轴，不参与压力去噪
    if ph_col is not None and ph_col in df.columns:
        ph_series = df[ph_col].interpolate().bfill().ffill().to_numpy(dtype=float)
        ph_uniform = np.interp(t_uniform, t_raw, ph_series)
    else:
        ph_uniform = np.full_like(t_uniform, np.nan, dtype=float)

    if temp_col is not None and temp_col in df.columns:
        temp_series = df[temp_col].interpolate().bfill().ffill().to_numpy(dtype=float)
        temp_uniform = np.interp(t_uniform, t_raw, temp_series)
    else:
        temp_uniform = np.full_like(t_uniform, np.nan, dtype=float)

    # ============================================================
    # Step 7. 生理事件保护型自适应平滑
    # ============================================================
    pressure_clean = _event_preserving_smooth(
        t_uniform,
        pressure_preclean_uniform,
        physio_uniform,
        event_score_uniform,
        smooth_window_s=smooth_window_s
    )

    residual_uniform = pressure_clean - baseline_uniform

    quality_flag = np.array(["good"] * len(t_uniform), dtype=object)
    quality_flag[artifact_uniform] = "artifact_reconstructed"
    quality_flag[range_abnormal_uniform] = "hard_range_reconstructed"
    quality_flag[low_conf_uniform] = "low_confidence"

    df_out = pd.DataFrame({
        "Time": t_uniform,
        "PH": ph_uniform,
        "Temperature": temp_uniform,

        # 完全原始压力
        "Pressure_Raw": pressure_raw_uniform,

        # 只经过 85–130 硬阈值修复后的压力
        "Pressure_RangeFixed": pressure_range_fixed_uniform,

        # 经过硬阈值 + 自适应伪影重建后的压力
        "Pressure_Preclean": pressure_preclean_uniform,

        # 最终平滑后的压力
        "Pressure_Clean": pressure_clean,

        "Baseline": baseline_uniform,
        "Residual": residual_uniform,
        "LocalNoiseSigma": sigma_uniform,

        # 新增：原始压力是否超出 95–120
        "RangeAbnormalMask": range_abnormal_uniform,

        # 包含 RangeAbnormalMask 和后续形态学伪影
        "ArtifactMask": artifact_uniform,

        "PhysioEventMask": physio_uniform,
        "QualityFlag": quality_flag
    })

    return df_out

#===========================================================================================================================================
def reconstruct_from_selected_modes(vmd_result, reference_signal, selected_mode_numbers):
    """
    根据指定 Mode 编号重构信号。
    自动补回均值，避免 VMD 模态和变成 0 均值信号。
    """
    selected_idx = [m - 1 for m in selected_mode_numbers]

    modes = np.asarray(vmd_result.modes, dtype=float)
    reference_signal = np.asarray(reference_signal, dtype=float)

    n = min(modes.shape[1], len(reference_signal))

    modes = modes[:, :n]
    reference_signal = reference_signal[:n]

    # VMD modes 通常是围绕 0 的振荡成分，需要补回压力均值
    reconstructed = np.sum(modes[selected_idx, :], axis=0)

    # 如果重构信号均值明显偏离参考信号，补回均值
    reconstructed = reconstructed - np.nanmean(reconstructed) + np.nanmean(reference_signal)

    return reconstructed


def compute_reconstruction_metrics(reference, reconstructed):
    reference = np.asarray(reference, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)

    n = min(len(reference), len(reconstructed))
    reference = reference[:n]
    reconstructed = reconstructed[:n]

    valid = np.isfinite(reference) & np.isfinite(reconstructed)

    ref = reference[valid]
    rec = reconstructed[valid]

    rmse = np.sqrt(np.mean((rec - ref) ** 2))
    corr = np.corrcoef(ref, rec)[0, 1]

    return rmse, corr