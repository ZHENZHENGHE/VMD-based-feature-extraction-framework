# -*- coding: utf-8 -*-
"""
09_publication_confusion_matrix.py

我用这个脚本补充论文级 confusion matrix 图和分类指标表。

输入：
    results/ml_validation_loso/loso_ml_predictions.csv

输出：
    results/ml_validation_loso/publication_figures/
        confusion_matrix_event_fixed_RandomForest.png
        classification_metrics_event_fixed_RandomForest.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
)


ROOT_DIR = Path(r"D:/a_work/课题组实验数据处理/新预处理/results")
PRED_FILE = ROOT_DIR / "ml_validation_loso" / "loso_ml_predictions.csv"

OUT_DIR = ROOT_DIR / "ml_validation_loso" / "publication_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "RandomForest"

METHOD_ORDER = ["event_guided", "fixed"]
METHOD_LABELS = {
    "event_guided": "Event-guided",
    "fixed": "Fixed-window",
}

# 我这里约定：0=Healthy/Normal，1=STC/Patient。
CLASS_LABELS = ["Healthy", "STC"]


def find_column(df, candidates):
    """我自动匹配不同脚本可能产生的列名。"""

    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def load_predictions(path: Path):
    """我读取 LOSO prediction 表，并统一列名。"""

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    method_col = find_column(df, ["Method", "WindowMethod"])
    model_col = find_column(df, ["Model", "Classifier"])
    subject_col = find_column(df, ["SubjectID", "SID", "subject"])
    y_true_col = find_column(df, ["TrueLabel", "y_true", "Label"])
    y_pred_col = find_column(df, ["PredLabel", "y_pred", "Pred", "Prediction"])

    required = {
        "Method": method_col,
        "Model": model_col,
        "SubjectID": subject_col,
        "TrueLabel": y_true_col,
        "PredLabel": y_pred_col,
    }

    missing = [k for k, v in required.items() if v is None]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Existing columns: {df.columns.tolist()}"
        )

    out = df.rename(columns={
        method_col: "Method",
        model_col: "Model",
        subject_col: "SubjectID",
        y_true_col: "TrueLabel",
        y_pred_col: "PredLabel",
    }).copy()

    out["Method"] = out["Method"].astype(str)
    out["Model"] = out["Model"].astype(str)
    out["SubjectID"] = out["SubjectID"].astype(str)
    out["TrueLabel"] = out["TrueLabel"].astype(int)
    out["PredLabel"] = out["PredLabel"].astype(int)

    return out


def compute_metrics(y_true, y_pred):
    """我计算医学分类指标。"""

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    sen = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spe = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "ACC": acc,
        "SEN": sen,
        "SPE": spe,
        "F1": f1,
        "PPV": ppv,
        "NPV": npv,
    }


def plot_single_confusion_matrix(ax, cm, title):
    """我画单个混淆矩阵。"""

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_title(title, fontsize=15)
    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_yticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS, fontsize=12)
    ax.set_yticklabels(CLASS_LABELS, fontsize=12)

    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
                fontweight="bold",
            )

    return im


if __name__ == "__main__":

    df = load_predictions(PRED_FILE)

    print("Available models:")
    print(df[["Method", "Model"]].drop_duplicates().sort_values(["Method", "Model"]))

    sub_df = df[df["Model"] == MODEL_NAME].copy()

    if sub_df.empty:
        raise ValueError(
            f"Model {MODEL_NAME} not found. "
            f"Available models: {sorted(df['Model'].unique())}"
        )

    metrics_rows = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    ims = []

    for ax, method in zip(axes, METHOD_ORDER):

        method_df = sub_df[sub_df["Method"] == method].copy()

        if method_df.empty:
            raise ValueError(f"Method not found for {MODEL_NAME}: {method}")

        y_true = method_df["TrueLabel"].values
        y_pred = method_df["PredLabel"].values

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        metrics = compute_metrics(y_true, y_pred)
        metrics.update({
            "Method": method,
            "MethodLabel": METHOD_LABELS.get(method, method),
            "Model": MODEL_NAME,
            "N": len(method_df),
        })

        metrics_rows.append(metrics)

        title = (
            f"{METHOD_LABELS.get(method, method)}\n"
            f"ACC={metrics['ACC']:.3f}, SEN={metrics['SEN']:.3f}, SPE={metrics['SPE']:.3f}"
        )

        im = plot_single_confusion_matrix(ax, cm, title)
        ims.append(im)

    cbar = fig.colorbar(
        ims[0],
        ax=axes.ravel().tolist(),
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.tick_params(labelsize=11)

    plt.suptitle(
        f"LOSO confusion matrices ({MODEL_NAME})",
        fontsize=17,
        y=1.03,
    )

    plt.tight_layout()

    out_png = OUT_DIR / f"confusion_matrix_event_fixed_{MODEL_NAME}.png"
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()

    metrics_df = pd.DataFrame(metrics_rows)

    ordered_cols = [
        "Method",
        "MethodLabel",
        "Model",
        "N",
        "TN",
        "FP",
        "FN",
        "TP",
        "ACC",
        "SEN",
        "SPE",
        "F1",
        "PPV",
        "NPV",
    ]

    metrics_df = metrics_df[ordered_cols]

    out_csv = OUT_DIR / f"classification_metrics_event_fixed_{MODEL_NAME}.csv"
    metrics_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Saved:")
    print(out_png)
    print(out_csv)

    print("\nMetrics:")
    print(metrics_df)
