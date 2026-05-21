
# ============================================================
# 工具函数（预处理评价指标）
# ============================================================
import numpy as np
import pandas as pd


def get_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    segments = []

    i = 0
    n = len(mask)

    while i < n:
        if not mask[i]:
            i += 1
            continue

        start = i

        while i < n and mask[i]:
            i += 1

        end = i
        segments.append((start, end))

    return segments


def compute_physio_event_preservation_metrics(
    df_out,
    time_col="Time",
    baseline_col="Baseline",
    pre_col="Pressure_Preclean",
    clean_col="Pressure_Clean",
    event_col="PhysioEventMask",
    min_points=3
):
    t = df_out[time_col].to_numpy(dtype=float)
    baseline = df_out[baseline_col].to_numpy(dtype=float)
    pre = df_out[pre_col].to_numpy(dtype=float)
    clean = df_out[clean_col].to_numpy(dtype=float)
    event_mask = df_out[event_col].to_numpy(dtype=bool)

    rows = []

    for event_id, (s, e) in enumerate(get_segments(event_mask), start=1):
        if e - s < min_points:
            continue

        tt = t[s:e]
        bb = baseline[s:e]
        pp = pre[s:e]
        cc = clean[s:e]

        valid = (
            np.isfinite(tt) &
            np.isfinite(bb) &
            np.isfinite(pp) &
            np.isfinite(cc)
        )

        if valid.sum() < min_points:
            continue

        tt = tt[valid]
        bb = bb[valid]
        pp = pp[valid]
        cc = cc[valid]

        # 相对 baseline 的事件幅度
        pre_res = pp - bb
        clean_res = cc - bb

        # 只计算正向压力事件面积
        pre_pos = np.maximum(pre_res, 0)
        clean_pos = np.maximum(clean_res, 0)

        pre_peak_amp = np.max(pre_pos)
        clean_peak_amp = np.max(clean_pos)

        pre_area = np.trapz(pre_pos, tt)
        clean_area = np.trapz(clean_pos, tt)

        peak_ratio = clean_peak_amp / (pre_peak_amp + 1e-9)
        area_ratio = clean_area / (pre_area + 1e-9)

        # 形态相关性：越接近 1，说明形态越相似
        if np.std(pre_res) > 1e-9 and np.std(clean_res) > 1e-9:
            shape_corr = np.corrcoef(pre_res, clean_res)[0, 1]
        else:
            shape_corr = np.nan

        # 归一化误差：越小越好
        rmse = np.sqrt(np.mean((clean_res - pre_res) ** 2))
        nrmse = rmse / (np.max(pre_res) - np.min(pre_res) + 1e-9)

        rows.append({
            "EventID": event_id,
            "StartTime": tt[0],
            "EndTime": tt[-1],
            "Duration_s": tt[-1] - tt[0],

            "Preclean_PeakAmplitude": pre_peak_amp,
            "Clean_PeakAmplitude": clean_peak_amp,
            "PeakPreservationRatio": peak_ratio,

            "Preclean_Area": pre_area,
            "Clean_Area": clean_area,
            "AreaPreservationRatio": area_ratio,

            "ShapeCorrelation": shape_corr,
            "Shape_NRMSE": nrmse,

            "NumPoints": len(tt)
        })

    event_metrics = pd.DataFrame(rows)

    return event_metrics