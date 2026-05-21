
# ============================================================
# 工具函数（预处理评价指标clean）
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ============================================================
# 工具函数（预处理评价指标preclean）
# ============================================================

def compute_preclean_preservation_metrics(
    df_out,
    time_col="Time",
    baseline_col="Baseline",
    reference_col="Pressure_RangeFixed",
    preclean_col="Pressure_Preclean",
    event_col="PhysioEventMask",
    artifact_col="ArtifactMask",
    range_col="RangeAbnormalMask",
    min_points=3
):
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

            s = i

            while i < n and mask[i]:
                i += 1

            segments.append((s, i))

        return segments

    t = df_out[time_col].to_numpy(dtype=float)
    baseline = df_out[baseline_col].to_numpy(dtype=float)
    ref = df_out[reference_col].to_numpy(dtype=float)
    pre = df_out[preclean_col].to_numpy(dtype=float)

    event_mask = df_out[event_col].to_numpy(dtype=bool)
    artifact_mask = df_out[artifact_col].to_numpy(dtype=bool)
    range_mask = df_out[range_col].to_numpy(dtype=bool)

    rows = []

    for event_id, (s, e) in enumerate(get_segments(event_mask), start=1):

        if e - s < min_points:
            continue

        seg_artifact_ratio = artifact_mask[s:e].mean()
        seg_range_ratio = range_mask[s:e].mean()

        # 如果这个事件区域大量包含伪影点，就不适合用于“生理保留”评价
        if seg_artifact_ratio > 0.1 or seg_range_ratio > 0.1:
            continue

        tt = t[s:e]
        bb = baseline[s:e]
        rr = ref[s:e]
        pp = pre[s:e]

        valid = (
            np.isfinite(tt) &
            np.isfinite(bb) &
            np.isfinite(rr) &
            np.isfinite(pp)
        )

        if valid.sum() < min_points:
            continue

        tt = tt[valid]
        bb = bb[valid]
        rr = rr[valid]
        pp = pp[valid]

        ref_res = rr - bb
        pre_res = pp - bb

        ref_pos = np.maximum(ref_res, 0)
        pre_pos = np.maximum(pre_res, 0)

        ref_peak = np.max(ref_pos)
        pre_peak = np.max(pre_pos)

        ref_area = np.trapz(ref_pos, tt)
        pre_area = np.trapz(pre_pos, tt)

        peak_ratio = pre_peak / (ref_peak + 1e-9)
        area_ratio = pre_area / (ref_area + 1e-9)

        if np.std(ref_res) > 1e-9 and np.std(pre_res) > 1e-9:
            shape_corr = np.corrcoef(ref_res, pre_res)[0, 1]
        else:
            shape_corr = np.nan

        rmse = np.sqrt(np.mean((pre_res - ref_res) ** 2))
        nrmse = rmse / (np.max(ref_res) - np.min(ref_res) + 1e-9)

        rows.append({
            "EventID": event_id,
            "StartTime": tt[0],
            "EndTime": tt[-1],
            "Duration_s": tt[-1] - tt[0],

            "Reference_PeakAmplitude": ref_peak,
            "Preclean_PeakAmplitude": pre_peak,
            "PeakPreservationRatio": peak_ratio,

            "Reference_Area": ref_area,
            "Preclean_Area": pre_area,
            "AreaPreservationRatio": area_ratio,

            "ShapeCorrelation": shape_corr,
            "Shape_NRMSE": nrmse,

            "ArtifactRatioInEvent": seg_artifact_ratio,
            "RangeAbnormalRatioInEvent": seg_range_ratio
        })

    return pd.DataFrame(rows)


def robust_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def first_diff_energy(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    dx = np.diff(x)
    return np.sum(dx ** 2)


def compute_noise_reduction_summary(df_out):
    stable_mask = (
        (~df_out["ArtifactMask"]) &
        (~df_out["PhysioEventMask"]) &
        np.isfinite(df_out["Pressure_Raw"]) &
        np.isfinite(df_out["Pressure_RangeFixed"]) &
        np.isfinite(df_out["Pressure_Preclean"]) &
        np.isfinite(df_out["Baseline"])
    )

    rows = []

    for name, col in [
        ("Raw", "Pressure_Raw"),
        ("RangeFixed", "Pressure_RangeFixed"),
        ("Preclean", "Pressure_Preclean")
    ]:
        x = df_out.loc[stable_mask, col].to_numpy(dtype=float)
        baseline = df_out.loc[stable_mask, "Baseline"].to_numpy(dtype=float)

        residual = x - baseline
        dx = np.diff(x)

        rows.append({
            "Signal": name,
            "Std": np.nanstd(x),
            "Residual_MAD": robust_mad(residual),
            "FirstDiff_MAD": robust_mad(dx),
            "FirstDiffEnergy": first_diff_energy(x),
            "ValidPoints": len(x)
        })

    summary = pd.DataFrame(rows)

    raw_row = summary[summary["Signal"] == "Raw"].iloc[0]

    summary["Residual_MAD_Reduction_%"] = (
        100 * (raw_row["Residual_MAD"] - summary["Residual_MAD"])
        / (raw_row["Residual_MAD"] + 1e-9)
    )

    summary["FirstDiff_MAD_Reduction_%"] = (
        100 * (raw_row["FirstDiff_MAD"] - summary["FirstDiff_MAD"])
        / (raw_row["FirstDiff_MAD"] + 1e-9)
    )

    summary["FirstDiffEnergy_Reduction_%"] = (
        100 * (raw_row["FirstDiffEnergy"] - summary["FirstDiffEnergy"])
        / (raw_row["FirstDiffEnergy"] + 1e-9)
    )

    return summary

def compute_preservation_pass_rate(event_metrics):
    peak_ok = event_metrics["PeakPreservationRatio"].between(0.90, 1.10)
    area_ok = event_metrics["AreaPreservationRatio"].between(0.90, 1.10)
    shape_ok = event_metrics["ShapeCorrelation"] >= 0.90
    nrmse_ok = event_metrics["Shape_NRMSE"] <= 0.15

    event_metrics = event_metrics.copy()

    event_metrics["Peak_OK"] = peak_ok
    event_metrics["Area_OK"] = area_ok
    event_metrics["Shape_OK"] = shape_ok
    event_metrics["NRMSE_OK"] = nrmse_ok

    event_metrics["Overall_Preserved"] = (
        peak_ok &
        area_ok &
        shape_ok &
        nrmse_ok
    )

    preservation_rate_summary = pd.DataFrame({
        "Criterion": [
            "Peak ratio within 0.90–1.10",
            "Area ratio within 0.90–1.10",
            "Shape correlation ≥ 0.90",
            "Shape NRMSE ≤ 0.15",
            "All criteria satisfied"
        ],
        "PassedEvents": [
            peak_ok.sum(),
            area_ok.sum(),
            shape_ok.sum(),
            nrmse_ok.sum(),
            event_metrics["Overall_Preserved"].sum()
        ],
        "TotalEvents": [
            len(event_metrics),
            len(event_metrics),
            len(event_metrics),
            len(event_metrics),
            len(event_metrics)
        ],
        "PassRate": [
            peak_ok.mean(),
            area_ok.mean(),
            shape_ok.mean(),
            nrmse_ok.mean(),
            event_metrics["Overall_Preserved"].mean()
        ]
    })

    return preservation_rate_summary, event_metrics

def compute_vmd_input_readiness(df_out):
    rows = []

    for name, col in [
        ("Raw", "Pressure_Raw"),
        ("RangeFixed", "Pressure_RangeFixed"),
        ("Preclean", "Pressure_Preclean")
    ]:
        x = df_out[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]

        dx = np.diff(x)

        rows.append({
            "Signal": name,
            "Length": len(x),
            "Std": np.std(x),
            "FirstDiff_MAD": robust_mad(dx),
            "FirstDiffEnergy": np.sum(dx ** 2),
            "MaxAbsJump": np.max(np.abs(dx)) if len(dx) > 0 else np.nan
        })

    return pd.DataFrame(rows)

