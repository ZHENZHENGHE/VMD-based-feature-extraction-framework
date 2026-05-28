# -*- coding: utf-8 -*-
"""
07_phase_space_statistics.py

用这个脚本补充正式论文需要的三个部分：
1. PhaseSpace feature-space visualization：基于受试者级 PhaseSpace 特征做 PCA 可视化。
2. Bootstrap 95% CI：基于 LOSO 预测结果计算 ACC/SEN/SPE/F1/AUC 的置信区间。
3. DeLong test + McNemar test：比较 event-guided 与 fixed 的 AUC 和配对分类错误。

注意：这里的 PCA 图不是原始相空间轨迹图。原始 trajectory 需要 phase_df 和窗口索引，不能只靠 subject-level 表恢复。
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import norm

ROOT_DIR = Path(r"D:/a_work/课题组实验数据处理/新预处理/results")
MERGED_DIR = ROOT_DIR / "merged_ml"
VALID_DIR = ROOT_DIR / "ml_validation_loso"
OUT_DIR = VALID_DIR / "statistical_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_SUBJECT_FILE = MERGED_DIR / "all_event_subject_features.csv"
FIXED_SUBJECT_FILE = MERGED_DIR / "all_fixed_subject_features.csv"
PRED_FILE = VALID_DIR / "loso_ml_predictions.csv"

N_BOOTSTRAP = 2000
RANDOM_STATE = 42
PREFERRED_MODEL_NAMES = ["RandomForest", "RF", "Logistic_L2", "SVM_RBF"]


def compute_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "SEN": recall_score(y_true, y_pred, zero_division=0),
        "SPE": spe,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": auc,
    }


def bootstrap_ci(y_true, y_pred, y_score, n_bootstrap=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    rows = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        rows.append(compute_metrics(y_true[idx], y_pred[idx], y_score[idx]))
    boot_df = pd.DataFrame(rows)
    point = compute_metrics(y_true, y_pred, y_score)
    summary = []
    for metric in ["ACC", "SEN", "SPE", "F1", "AUC"]:
        values = boot_df[metric].dropna().values
        summary.append({
            "Metric": metric,
            "PointEstimate": point[metric],
            "CI_low": np.percentile(values, 2.5),
            "CI_high": np.percentile(values, 97.5),
            "N_bootstrap_valid": len(values),
        })
    return pd.DataFrame(summary), boot_df


def compute_midrank(x):
    x = np.asarray(x)
    order = np.argsort(x)
    sx = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sx[j] == sx[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_roc_test(y_true, score_a, score_b):
    y_true = np.asarray(y_true).astype(int)
    score_a = np.asarray(score_a).astype(float)
    score_b = np.asarray(score_b).astype(float)
    order = np.argsort(-y_true)
    m = int(np.sum(y_true))
    preds = np.vstack((score_a, score_b))[:, order]
    aucs, cov = fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    if np.ndim(cov) == 0:
        var = float(cov)
    else:
        var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z, p = np.nan, np.nan
    else:
        z = diff / np.sqrt(var)
        p = 2 * (1 - norm.cdf(abs(z)))
    return {"AUC_1": aucs[0], "AUC_2": aucs[1], "AUC_diff_1_minus_2": diff, "z": z, "p": p}


def load_subject_table(path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    y = df["Label"].astype(int).values
    sid = df["SubjectID"].astype(str).values
    X = df.drop(columns=["SubjectID", "Label"], errors="ignore").select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    return df, X, y, sid


def drop_non_biomarker_cols(X):
    bad_patterns = ["Tau", "EmbeddingDim", "EmbeddedVectors", "RQA_epsilon", "NumWindows", "WindowID", "StartTime", "EndTime", "CenterTime", "StartIndex", "EndIndex"]
    drop_cols = [c for c in X.columns if any(p in c for p in bad_patterns)]
    return X.drop(columns=drop_cols, errors="ignore"), drop_cols


def plot_phase_space_pca(event_X, event_y, event_sid, fixed_X, fixed_y, fixed_sid):
    rows = []
    for method, X, y, sid in [("event_guided", event_X, event_y, event_sid), ("fixed", fixed_X, fixed_y, fixed_sid)]:
        cols = [c for c in X.columns if "PS_" in c]
        X_ps = X[cols].copy()
        if X_ps.shape[1] < 2:
            print(f"Skip {method}: too few PhaseSpace features.")
            continue
        Z = SimpleImputer(strategy="median").fit_transform(X_ps)
        Z = StandardScaler().fit_transform(Z)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(Z)
        tmp = pd.DataFrame({"SubjectID": sid, "Label": y, "Method": method, "PC1": coords[:, 0], "PC2": coords[:, 1], "ExplainedVar_PC1": pca.explained_variance_ratio_[0], "ExplainedVar_PC2": pca.explained_variance_ratio_[1]})
        rows.append(tmp)
        plt.figure(figsize=(6, 5))
        for label_value, label_name, marker in [(0, "healthy", "o"), (1, "patient", "^")]:
            mask = y == label_value
            plt.scatter(coords[mask, 0], coords[mask, 1], label=label_name, marker=marker, s=70, alpha=0.85)
        for i, s in enumerate(sid):
            plt.text(coords[i, 0], coords[i, 1], str(s), fontsize=7, alpha=0.7)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.title(f"{method}: PhaseSpace feature PCA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{method}_phase_space_feature_pca.png", dpi=300)
        plt.close()
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(OUT_DIR / "phase_space_feature_pca_coordinates.csv", index=False, encoding="utf-8-sig")


def find_column(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}\n请先运行 05_loso_robust_validation.py。")
    df = pd.read_csv(path)
    method_col = find_column(df, ["Method", "WindowMethod"])
    model_col = find_column(df, ["Model", "Classifier"])
    sid_col = find_column(df, ["SubjectID", "SID", "subject"])
    y_col = find_column(df, ["y_true", "Label", "TrueLabel"])
    pred_col = find_column(df, ["y_pred", "Pred", "Prediction", "PredLabel"])
    score_col = find_column(df, ["y_score", "Score", "Prob", "Probability", "y_prob"])
    required = {"Method": method_col, "Model": model_col, "SubjectID": sid_col, "y_true": y_col, "y_pred": pred_col, "y_score": score_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Prediction table missing columns: {missing}\nExisting columns: {df.columns.tolist()}")
    out = df.rename(columns={method_col: "Method", model_col: "Model", sid_col: "SubjectID", y_col: "y_true", pred_col: "y_pred", score_col: "y_score"}).copy()
    out["Method"] = out["Method"].astype(str)
    out["Model"] = out["Model"].astype(str)
    out["SubjectID"] = out["SubjectID"].astype(str)
    out["y_true"] = out["y_true"].astype(int)
    out["y_pred"] = out["y_pred"].astype(int)
    out["y_score"] = out["y_score"].astype(float)
    return out


def choose_model(pred_df):
    print("\nAvailable models:")
    print(pred_df[["Method", "Model"]].drop_duplicates().sort_values(["Method", "Model"]))
    for name in PREFERRED_MODEL_NAMES:
        if name in pred_df["Model"].unique():
            print(f"\nUse preferred model: {name}")
            return name
    name = pred_df["Model"].iloc[0]
    print(f"\nUse first available model: {name}")
    return name


def run_bootstrap_for_all(pred_df):
    rows = []
    boot_rows = []
    for (method, model), sub in pred_df.groupby(["Method", "Model"]):
        ci, boot = bootstrap_ci(sub["y_true"].values, sub["y_pred"].values, sub["y_score"].values, N_BOOTSTRAP, RANDOM_STATE)
        ci["Method"] = method
        ci["Model"] = model
        boot["Method"] = method
        boot["Model"] = model
        rows.append(ci)
        boot_rows.append(boot)
    ci_all = pd.concat(rows, ignore_index=True)
    boot_all = pd.concat(boot_rows, ignore_index=True)
    ci_all.to_csv(OUT_DIR / "bootstrap_95ci_summary.csv", index=False, encoding="utf-8-sig")
    boot_all.to_csv(OUT_DIR / "bootstrap_metric_distribution.csv", index=False, encoding="utf-8-sig")
    return ci_all


def run_event_vs_fixed_tests(pred_df, model_name):
    sub = pred_df[pred_df["Model"] == model_name].copy()
    if not {"event_guided", "fixed"}.issubset(set(sub["Method"].unique())):
        raise ValueError("Prediction table must contain both event_guided and fixed methods.")
    event_df = sub[sub["Method"] == "event_guided"].copy()
    fixed_df = sub[sub["Method"] == "fixed"].copy()
    merged = event_df.merge(fixed_df, on=["SubjectID", "y_true"], suffixes=("_event", "_fixed"))
    if merged.empty:
        raise ValueError("No paired subjects found between event_guided and fixed.")
    y_true = merged["y_true"].values
    delong = delong_roc_test(y_true, merged["y_score_event"].values, merged["y_score_fixed"].values)
    delong_df = pd.DataFrame([{ "Model": model_name, "Comparison": "event_guided_minus_fixed", **delong }])
    event_correct = merged["y_pred_event"].values == y_true
    fixed_correct = merged["y_pred_fixed"].values == y_true
    b = int(np.sum(event_correct & (~fixed_correct)))
    c = int(np.sum((~event_correct) & fixed_correct))
    result = mcnemar([[0, b], [c, 0]], exact=True)
    mcnemar_df = pd.DataFrame([{ "Model": model_name, "b_event_correct_fixed_wrong": b, "c_event_wrong_fixed_correct": c, "statistic": result.statistic, "p": result.pvalue }])
    delong_df.to_csv(OUT_DIR / f"delong_event_vs_fixed_{model_name}.csv", index=False, encoding="utf-8-sig")
    mcnemar_df.to_csv(OUT_DIR / f"mcnemar_event_vs_fixed_{model_name}.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(OUT_DIR / f"paired_predictions_event_vs_fixed_{model_name}.csv", index=False, encoding="utf-8-sig")
    return delong_df, mcnemar_df


if __name__ == "__main__":
    print("=" * 70)
    print("1. PhaseSpace feature-space visualization")
    print("=" * 70)
    _, event_X, event_y, event_sid = load_subject_table(EVENT_SUBJECT_FILE)
    _, fixed_X, fixed_y, fixed_sid = load_subject_table(FIXED_SUBJECT_FILE)
    event_X, event_dropped = drop_non_biomarker_cols(event_X)
    fixed_X, fixed_dropped = drop_non_biomarker_cols(fixed_X)
    pd.DataFrame({"Method": ["event_guided"] * len(event_dropped) + ["fixed"] * len(fixed_dropped), "DroppedFeature": event_dropped + fixed_dropped}).to_csv(OUT_DIR / "dropped_non_biomarker_features_for_stats.csv", index=False, encoding="utf-8-sig")
    plot_phase_space_pca(event_X, event_y, event_sid, fixed_X, fixed_y, fixed_sid)
    print("PhaseSpace PCA figures saved.")

    print("\n" + "=" * 70)
    print("2. Bootstrap CI")
    print("=" * 70)
    pred_df = load_predictions(PRED_FILE)
    ci_all = run_bootstrap_for_all(pred_df)
    print(ci_all)

    print("\n" + "=" * 70)
    print("3. DeLong test and McNemar test")
    print("=" * 70)
    model_name = choose_model(pred_df)
    delong_df, mcnemar_df = run_event_vs_fixed_tests(pred_df, model_name)
    print("\nDeLong test:")
    print(delong_df)
    print("\nMcNemar test:")
    print(mcnemar_df)
    print("\nFinished.")
    print("Saved to:", OUT_DIR.resolve())
