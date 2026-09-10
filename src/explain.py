import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

try:
    from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE
    from src.data import split_dataset
except ModuleNotFoundError:  # pragma: no cover
    from config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE
    from data import split_dataset


def main():
    X_train, X_test, y_train, y_test, scaler = split_dataset()
    model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

    sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
    sample = sample.copy()
    sample.columns = feature_names
    sample = sample.reset_index(drop=True)

    explainer = shap.Explainer(model, sample)
    shap_values = explainer(sample)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranking = sorted(zip(sample.columns, mean_abs_shap), key=lambda kv: kv[1], reverse=True)
    print("Top features by mean |SHAP value| (most to least important):")
    for rank, (feature, value) in enumerate(ranking[:5], start=1):
        print(f"{rank}. {feature}: {value:.4f}")

    output_path = FIGURES_DIR / "shap_summary.png"
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP summary plot to {output_path}")


if __name__ == "__main__":
    main()
