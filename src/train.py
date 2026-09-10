import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

try:
    from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TARGET
    from src.data import split_dataset
except ModuleNotFoundError:  # pragma: no cover
    from config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TARGET
    from data import split_dataset

# Fixed categorical colors: same slot always means the same model, never re-cycled.
COLOR_LR = "#2a78d6"     # blue
COLOR_XGB = "#eb6834"    # orange
COLOR_RF = "#1baf7a"     # aqua
COLOR_LGBM = "#eda100"   # yellow
COLOR_ISO = "#e87ba4"    # magenta
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f5f9ff", "#2a78d6"])

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "isolation_forest": "Isolation Forest (unsupervised)",
}
MODEL_COLORS = {
    "logistic_regression": COLOR_LR,
    "xgboost": COLOR_XGB,
    "random_forest": COLOR_RF,
    "lightgbm": COLOR_LGBM,
    "isolation_forest": COLOR_ISO,
}


def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }
    return metrics


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix:")
    print(cm)
    print(f"Fraud caught: {tp} / Actual fraud")
    print(f"Fraud missed: {fn} / Actual fraud")
    print(f"Legit flagged as fraud: {fp}")


def save_confusion_matrices(results, output_path: Path):
    n = len(results)
    cols = min(n, 3)
    rows = -(-n // cols)  # ceil division
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (name, y_true, y_pred) in zip(axes, results):
        cm = confusion_matrix(y_true, y_pred)
        ax.imshow(cm, cmap=SEQUENTIAL_BLUE)
        ax.set_title(name)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted legit", "Predicted fraud"])
        ax.set_yticklabels(["Actual legit", "Actual fraud"])
        vmax = cm.max()
        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                text_color = "white" if value > vmax * 0.6 else "#1a1a19"
                ax.text(j, i, f"{value:,}", ha="center", va="center", color=text_color, fontsize=13)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Confusion Matrix: Fraud Caught vs. Missed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_pr_curves(y_test, model_probs, fraud_rate, output_path: Path):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for name, y_prob, color in model_probs:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=name)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ax_pr.plot(recall, precision, color=color, linewidth=2, label=name)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#9a9a94", linewidth=1, label="Chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right", frameon=False)

    ax_pr.axhline(fraud_rate, linestyle="--", color="#9a9a94", linewidth=1, label="Baseline (fraud rate)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(loc="center left", bbox_to_anchor=(0.02, 0.45), frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics_comparison(all_metrics, colors, output_path: Path):
    metric_keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    names = list(all_metrics.keys())
    n = len(names)

    x = np.arange(len(metric_keys))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(9, 2.2 * n + 3), 5))
    for i, name in enumerate(names):
        values = [all_metrics[name][k] for k in metric_keys]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color=colors[name])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#52514e")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (accuracy intentionally omitted)")
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_comparison_table(all_metrics):
    rows = sorted(all_metrics.items(), key=lambda kv: kv[1]["pr_auc"], reverse=True)
    header = f"{'Model':<32}{'Precision':>10}{'Recall':>10}{'F1':>10}{'ROC-AUC':>10}{'PR-AUC':>10}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(
            f"{name:<32}{m['precision']:>10.3f}{m['recall']:>10.3f}"
            f"{m['f1']:>10.3f}{m['roc_auc']:>10.3f}{m['pr_auc']:>10.3f}"
        )


def train_and_evaluate():
    X_train, X_test, y_train, y_test, scaler = split_dataset()

    feature_names = list(X_train.columns)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_metrics = calculate_metrics(y_test, lr_pred, lr_prob)

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        random_state=RANDOM_STATE,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = calculate_metrics(y_test, xgb_pred, xgb_prob)

    print("Logistic Regression metrics:")
    print(json.dumps(lr_metrics, indent=2))
    print_confusion_matrix(y_test, lr_pred)

    print("\nXGBoost metrics:")
    print(json.dumps(xgb_metrics, indent=2))
    print_confusion_matrix(y_test, xgb_pred)

    rf = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        max_depth=16,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_metrics = calculate_metrics(y_test, rf_pred, rf_prob)

    print("\nRandom Forest metrics:")
    print(json.dumps(rf_metrics, indent=2))
    print_confusion_matrix(y_test, rf_pred)

    # scale_pos_weight=1 (no reweighting): PR-AUC is a ranking metric, and reweighting
    # toward the minority class degrades LightGBM's ranking here rather than helping it
    # (see the README's LightGBM section for the sweep evidence). Imbalance is instead
    # handled downstream, at the decision threshold (the app's threshold slider).
    lgbm = LGBMClassifier(
        scale_pos_weight=1,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_test)
    lgbm_prob = lgbm.predict_proba(X_test)[:, 1]  # probability of class 1 (fraud) — used for ROC-AUC/PR-AUC
    lgbm_metrics = calculate_metrics(y_test, lgbm_pred, lgbm_prob)

    print("\nLightGBM metrics:")
    print(json.dumps(lgbm_metrics, indent=2))
    print_confusion_matrix(y_test, lgbm_pred)

    # Isolation Forest is unsupervised: it never sees y_train, only the feature matrix.
    iso_contamination = float(y_train.mean())
    iso = IsolationForest(
        contamination=iso_contamination,
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train)
    iso_raw_pred = iso.predict(X_test)  # -1 = anomaly, 1 = normal
    iso_pred = np.where(iso_raw_pred == -1, 1, 0)
    iso_prob = -iso.score_samples(X_test)  # negate so higher = more fraud-like
    iso_metrics = calculate_metrics(y_test, iso_pred, iso_prob)

    print(f"\nIsolation Forest metrics (unsupervised, contamination={iso_contamination:.4%}):")
    print(json.dumps(iso_metrics, indent=2))
    print_confusion_matrix(y_test, iso_pred)

    all_metrics = {
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
        "random_forest": rf_metrics,
        "lightgbm": lgbm_metrics,
        "isolation_forest": iso_metrics,
    }
    all_preds = {
        "logistic_regression": lr_pred,
        "xgboost": xgb_pred,
        "random_forest": rf_pred,
        "lightgbm": lgbm_pred,
        "isolation_forest": iso_pred,
    }
    all_probs = {
        "logistic_regression": lr_prob,
        "xgboost": xgb_prob,
        "random_forest": rf_prob,
        "lightgbm": lgbm_prob,
        "isolation_forest": iso_prob,
    }

    print("\nModel comparison (sorted by PR-AUC):")
    print_comparison_table({DISPLAY_NAMES[k]: v for k, v in all_metrics.items()})

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fraud_rate = float(y_test.sum() / len(y_test))
    save_confusion_matrices(
        [(DISPLAY_NAMES[k], y_test, all_preds[k]) for k in all_metrics],
        FIGURES_DIR / "confusion_matrices.png",
    )
    save_roc_pr_curves(
        y_test,
        [(DISPLAY_NAMES[k], all_probs[k], MODEL_COLORS[k]) for k in all_metrics],
        fraud_rate,
        FIGURES_DIR / "roc_pr_curves.png",
    )
    save_metrics_comparison(
        {DISPLAY_NAMES[k]: v for k, v in all_metrics.items()},
        {DISPLAY_NAMES[k]: MODEL_COLORS[k] for k in all_metrics},
        FIGURES_DIR / "metrics_comparison.png",
    )
    print(f"\nSaved result visualizations to {FIGURES_DIR}")

    # Isolation Forest is unsupervised and not a deployment candidate here — it's a
    # comparison point for when labeled fraud examples aren't available.
    supervised_models = {
        "logistic_regression": lr,
        "xgboost": xgb,
        "random_forest": rf,
        "lightgbm": lgbm,
    }
    best_name = max(supervised_models, key=lambda k: all_metrics[k]["pr_auc"])
    best_model = supervised_models[best_name]
    print(
        f"\nBest supervised model by PR-AUC: {DISPLAY_NAMES[best_name]} "
        f"(pr_auc={all_metrics[best_name]['pr_auc']:.4f}) -> saved as the deployable model"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "xgboost_model.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    metrics = {
        **all_metrics,
        "best_supervised_model": best_name,
    }
    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        **all_metrics,
        "feature_names": feature_names,
        "best_supervised_model": best_name,
    }


if __name__ == "__main__":
    train_and_evaluate()
