import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

    output_path = FIGURES_DIR / "shap_summary.png"
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP summary plot to {output_path}")


if __name__ == "__main__":
    main()
