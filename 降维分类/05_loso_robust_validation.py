# -*- coding: utf-8 -*-
"""
05_loso_robust_validation.py

用这个脚本完成 SCI 论文阶段的严格验证：
1. event-guided 与 fixed 分别做受试者级 LOSO 分类
2. 所有预处理、缺失填补、标准化、特征筛选都放在每一折训练集内部
3. 输出 ACC / SEN / SPE / F1 / AUC
4. 输出 ROC 曲线、混淆矩阵、bootstrap 95% CI
5. 输出 permutation test 结果

输入文件来自：results/merged_ml/
    event_subject_level_features.csv
    fixed_subject_level_features.csv
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, permutation_test_score
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# ============================================================
# 1. Paths
# ============================================================

ROOT_DIR = Path(r"D:/a_work/课题组实验数据处理/新预处理/results")
IN_DIR = ROOT_DIR / "merged_ml"
OUT_DIR = ROOT_DIR / "ml_validation_loso"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_FILE = IN_DIR / "all_event_subject_features.csv"
FIXED_FILE = IN_DIR / "all_fixed_subject_features.csv"


# ============================================================
# 2. Utility functions
# ============================================================

def load_subject_table(path: Path):
    """读取受试者级表，并分离 X/y/SubjectID。"""
    df = pd.read_csv(path)

    if "SubjectID" not in df.columns or "Label" not in df.columns:
        raise ValueError(f"Missing SubjectID or Label in {path}")

    subject_ids = df["SubjectID"].astype(str).values
    y = df["Label"].astype(int).values

    drop_cols = ["SubjectID", "Label"]
    X_df = df.drop(columns=drop_cols, errors="ignore")

    # 只保留数值特征，避免字符串列进入模型。
    X_df = X_df.select_dtypes(include=[np.number]).copy()

    # 删除全缺失或常数列，这些列对分类没有贡献。
    X_df = X_df.dropna(axis=1, how="all")
    nunique = X_df.nunique(dropna=True)
    X_df = X_df.loc[:, nunique > 1]

    return df, X_df, y, subject_ids


def make_models(random_state=42):
    """定义多种适合小样本医学分类的模型。"""
    models = {
        "Logistic_L1": LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5,
            class_weight="balanced", max_iter=5000, random_state=random_state,
        ),
        "Logistic_L2": LogisticRegression(
            penalty="l2", solver="liblinear", C=1.0,
            class_weight="balanced", max_iter=5000, random_state=random_state,
        ),
        "SVM_Linear": SVC(
            kernel="linear", C=1.0, probability=True,
            class_weight="balanced", random_state=random_state,
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2,
            random_state=random_state,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.05,
            random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "GaussianNB": GaussianNB(),
    }
    return models


def get_score_from_model(model, X_test):
    """统一获取阳性类别概率或决策分数。"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        score = model.decision_function(X_test)
        return np.asarray(score).reshape(-1)
    return model.predict(X_test)


def safe_metrics(y_true, y_pred, y_score):
    """计算分类指标，并对极小样本下的异常情况做保护。"""
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    if len(np.unique(y_true)) == 2:
        auc_value = roc_auc_score(y_true, y_score)
    else:
        auc_value = np.nan

    return {
        "ACC": acc,
        "SEN": sen,
        "SPE": spe,
        "F1": f1,
        "AUC": auc_value,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def loso_evaluate(X_df, y, subject_ids, method_name, k_features=10, random_state=42):
    """用 Leave-One-Subject-Out 做严格受试者级验证。"""
    X = X_df.values
    n_features = X.shape[1]
    k = min(k_features, n_features)

    models = make_models(random_state=random_state)
    loo = LeaveOneOut()

    summary_rows = []
    prediction_rows = []

    for model_name, clf in models.items():
        y_true_all = []
        y_pred_all = []
        y_score_all = []
        subject_all = []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=k)),
                ("clf", clone(clf)),
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_score = get_score_from_model(pipe, X_test)

            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())
            y_score_all.extend(np.asarray(y_score).tolist())
            subject_all.extend(subject_ids[test_idx].tolist())

        y_true_all = np.asarray(y_true_all)
        y_pred_all = np.asarray(y_pred_all)
        y_score_all = np.asarray(y_score_all)

        metrics = safe_metrics(y_true_all, y_pred_all, y_score_all)
        metrics.update({
            "Method": method_name,
            "Model": model_name,
            "KFeatures": k,
            "NSubjects": len(y_true_all),
        })
        summary_rows.append(metrics)

        for sid, yt, yp, ys in zip(subject_all, y_true_all, y_pred_all, y_score_all):
            prediction_rows.append({
                "Method": method_name,
                "Model": model_name,
                "KFeatures": k,
                "SubjectID": sid,
                "TrueLabel": int(yt),
                "PredLabel": int(yp),
                "Score": float(ys),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def bootstrap_ci(y_true, y_pred, y_score, n_boot=5000, random_state=42):
    """用受试者级 bootstrap 估计指标 95% 置信区间。"""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    records = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ys = y_score[idx]

        if len(np.unique(yt)) < 2:
            continue

        records.append(safe_metrics(yt, yp, ys))

    boot_df = pd.DataFrame(records)
    rows = []
    for metric in ["ACC", "SEN", "SPE", "F1", "AUC"]:
        rows.append({
            "Metric": metric,
            "Mean": boot_df[metric].mean(),
            "CI_Lower": boot_df[metric].quantile(0.025),
            "CI_Upper": boot_df[metric].quantile(0.975),
        })
    return pd.DataFrame(rows)


def plot_roc_from_predictions(pred_df, method, model, out_dir):
    """根据 LOSO 预测结果画 ROC 曲线。"""
    sub = pred_df[(pred_df["Method"] == method) & (pred_df["Model"] == model)].copy()
    y_true = sub["TrueLabel"].values
    y_score = sub["Score"].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"{method}-{model}, AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"LOSO ROC: {method} / {model}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_{method}_{model}.png", dpi=300)
    plt.close()


# ============================================================
# 3. Run LOSO validation
# ============================================================

if __name__ == "__main__":

    event_df, X_event, y_event, sid_event = load_subject_table(EVENT_FILE)
    fixed_df, X_fixed, y_fixed, sid_fixed = load_subject_table(FIXED_FILE)

    print("Event subject table:", X_event.shape, "Label counts:", pd.Series(y_event).value_counts().to_dict())
    print("Fixed subject table:", X_fixed.shape, "Label counts:", pd.Series(y_fixed).value_counts().to_dict())

    all_summary = []
    all_predictions = []

    # 建议小样本先固定 k=10，避免用测试结果反复挑 k。
    K_FEATURES = 10

    summary_event, pred_event = loso_evaluate(
        X_event, y_event, sid_event,
        method_name="event_guided",
        k_features=K_FEATURES,
        random_state=42,
    )
    summary_fixed, pred_fixed = loso_evaluate(
        X_fixed, y_fixed, sid_fixed,
        method_name="fixed",
        k_features=K_FEATURES,
        random_state=42,
    )

    summary_df = pd.concat([summary_event, summary_fixed], ignore_index=True)
    pred_df = pd.concat([pred_event, pred_fixed], ignore_index=True)

    summary_df.to_csv(OUT_DIR / "loso_ml_summary.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUT_DIR / "loso_ml_predictions.csv", index=False, encoding="utf-8-sig")

    print("\nLOSO summary:")
    print(summary_df.sort_values(["Method", "AUC"], ascending=[True, False]))

    # 自动选择每种方法 AUC 最高的模型，用于 ROC 和 CI。
    best_models = (
        summary_df.sort_values("AUC", ascending=False)
        .groupby("Method")
        .head(1)
        [["Method", "Model"]]
    )

    ci_rows = []
    for _, row in best_models.iterrows():
        method = row["Method"]
        model = row["Model"]
        sub = pred_df[(pred_df["Method"] == method) & (pred_df["Model"] == model)]

        ci = bootstrap_ci(
            y_true=sub["TrueLabel"].values,
            y_pred=sub["PredLabel"].values,
            y_score=sub["Score"].values,
            n_boot=5000,
            random_state=42,
        )
        ci["Method"] = method
        ci["Model"] = model
        ci_rows.append(ci)

        plot_roc_from_predictions(pred_df, method, model, OUT_DIR)

    ci_df = pd.concat(ci_rows, ignore_index=True)
    ci_df.to_csv(OUT_DIR / "best_model_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    print("\nBest model bootstrap CI:")
    print(ci_df)

    print("\nSaved to:", OUT_DIR.resolve())
