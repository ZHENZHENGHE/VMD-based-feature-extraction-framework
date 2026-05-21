
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

import numpy as np
import pandas as pd

# ============================================================
# 1. 单次 VMD 参数评价函数
# ============================================================

def evaluate_single_vmd_setting(
    signal,
    time,
    event_mask,
    artifact_mask,
    baseline,
    K,
    alpha,
    min_keep_score=0.18
):
    """
    对单个 K-alpha 参数组合进行 VMD 分解、模态筛选、重构和评价。
    """

    # ---------- VMD + 生理约束模态筛选 ----------
    vmd_res = physiology_constrained_vmd_reconstruction(
        signal=signal,
        time=time,
        event_mask=event_mask,
        artifact_mask=artifact_mask,
        K=K,
        alpha=alpha,
        min_keep_score=min_keep_score
    )

    reconstructed = np.asarray(vmd_res.reconstructed, dtype=float)
    reference = np.asarray(signal, dtype=float)

    # ---------- 对齐长度 ----------
    n = min(
        len(reference),
        len(reconstructed),
        len(time),
        len(event_mask),
        len(baseline)
    )

    reference = reference[:n]
    reconstructed = reconstructed[:n]
    time = np.asarray(time[:n], dtype=float)
    event_mask = np.asarray(event_mask[:n], dtype=bool)
    baseline = np.asarray(baseline[:n], dtype=float)

    # ---------- 补回均值，避免 VMD 重构基线偏移 ----------
    reconstructed = (
        reconstructed
        - np.nanmean(reconstructed)
        + np.nanmean(reference)
    )

    valid = np.isfinite(reference) & np.isfinite(reconstructed)

    ref = reference[valid]
    rec = reconstructed[valid]

    rmse = np.sqrt(np.mean((rec - ref) ** 2))

    if np.std(ref) > 1e-9 and np.std(rec) > 1e-9:
        corr = np.corrcoef(ref, rec)[0, 1]
    else:
        corr = np.nan

    # ---------- 事件保真指标 ----------
    event_summary, event_detail = compute_event_fidelity_metrics(
        time=time,
        reference=reference,
        reconstructed=reconstructed,
        event_mask=event_mask,
        baseline=baseline
    )

    # ---------- 模态评分表 ----------
    mode_summary = summarize_physiology_vmd_result(vmd_res)

    selected_modes = mode_summary.loc[
        mode_summary["Selected"], "Mode"
    ].tolist()

    num_selected_modes = len(selected_modes)

    # ---------- 组合评分 ----------
    # RMSE需要归一化，否则量纲不同
    signal_range = np.nanmax(reference) - np.nanmin(reference) + 1e-9
    rmse_norm = rmse / signal_range

    peak_error = abs(event_summary["Mean_PeakPreservationRatio"] - 1.0)
    area_error = abs(event_summary["Mean_AreaPreservationRatio"] - 1.0)

    # 综合得分：越高越好
    objective_score = (
        0.25 * (1.0 - rmse_norm)
        + 0.25 * corr
        + 0.20 * event_summary["Mean_Event_IoU"]
        + 0.15 * (1.0 - peak_error)
        + 0.15 * (1.0 - area_error)
    )

    return {
        "K": K,
        "alpha": alpha,
        "min_keep_score": min_keep_score,

        "RMSE": rmse,
        "RMSE_Norm": rmse_norm,
        "Correlation": corr,
        "Mean_Event_IoU": event_summary["Mean_Event_IoU"],
        "Mean_PeakPreservationRatio": event_summary["Mean_PeakPreservationRatio"],
        "Mean_AreaPreservationRatio": event_summary["Mean_AreaPreservationRatio"],
        "NumEvents": event_summary["NumEvents"],

        "NumSelectedModes": num_selected_modes,
        "SelectedModes": selected_modes,

        "ObjectiveScore": objective_score
    }


# ============================================================
# 2. 网格搜索 K-alpha 最优组合
# ============================================================

def search_best_vmd_parameters(
    pred_df,
    signal_col="Pressure_Clean",
    K_list=None,
    alpha_list=None,
    min_keep_score=0.18
):
    """
    在指定 K 和 alpha 范围内搜索最优 VMD 参数组合。
    """

    if K_list is None:
        K_list = [3, 4, 5, 6, 7]

    if alpha_list is None:
        alpha_list = [1500, 2000, 2500, 2700, 3000, 3500]

    signal = pred_df[signal_col].values
    time = pred_df["Time"].values
    event_mask = pred_df["PhysioEventMask"].values
    artifact_mask = pred_df["ArtifactMask"].values
    baseline = pred_df["Baseline"].values

    rows = []

    for K in K_list:
        for alpha in alpha_list:
            try:
                result = evaluate_single_vmd_setting(
                    signal=signal,
                    time=time,
                    event_mask=event_mask,
                    artifact_mask=artifact_mask,
                    baseline=baseline,
                    K=K,
                    alpha=alpha,
                    min_keep_score=min_keep_score
                )

                rows.append(result)

                print(
                    f"Done: K={K}, alpha={alpha}, "
                    f"Score={result['ObjectiveScore']:.4f}, "
                    f"RMSE={result['RMSE']:.4f}, "
                    f"Corr={result['Correlation']:.4f}, "
                    f"Modes={result['SelectedModes']}"
                )

            except Exception as e:
                print(f"Failed: K={K}, alpha={alpha}, error={e}")

    result_df = pd.DataFrame(rows)

    result_df = result_df.sort_values(
        by="ObjectiveScore",
        ascending=False
    ).reset_index(drop=True)

    return result_df