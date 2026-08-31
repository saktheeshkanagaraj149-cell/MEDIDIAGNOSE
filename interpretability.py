import os
import numpy as np
import pandas as pd
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from preprocessing import (
    preprocess_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    SYMPTOM_FEATURES, DISEASE_LABELS
)


def run_shap_analysis(model_path="models/best_model.pkl", n_samples=200):
    print("Loading model and data...")
    model = joblib.load(model_path)
    feature_names = joblib.load("models/feature_names.pkl")
    X_train, X_test, y_train, y_test, _ = preprocess_data()

    X_sample = X_test[:n_samples]

    os.makedirs("outputs/shap", exist_ok=True)

    for i, disease in enumerate(DISEASE_LABELS):
        print(f"  SHAP for {disease}...")
        estimator = model.estimators_[i]

        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_sample)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample,
                            feature_names=feature_names,
                            show=False, max_display=15)
            plt.title(f"SHAP Summary - {disease}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"outputs/shap/shap_summary_{disease}.png",
                       dpi=150, bbox_inches="tight")
            plt.close()

            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample,
                            feature_names=feature_names,
                            plot_type="bar", show=False, max_display=15)
            plt.title(f"SHAP Feature Importance - {disease}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"outputs/shap/shap_importance_{disease}.png",
                       dpi=150, bbox_inches="tight")
            plt.close()

        except Exception as e:
            print(f"    Warning: SHAP failed for {disease}: {e}")

    print("SHAP analysis complete. Results saved to outputs/shap/")


def run_lime_analysis(model_path="models/best_model.pkl", n_samples=50):
    print("Loading model and data...")
    model = joblib.load(model_path)
    feature_names = joblib.load("models/feature_names.pkl")
    X_train, X_test, y_train, y_test, _ = preprocess_data()

    all_features = NUMERICAL_FEATURES.copy()
    cat_encoder = joblib.load("models/preprocessor").named_transformers_["cat"].named_steps["encoder"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_features.extend(cat_names)
    all_features.extend(SYMPTOM_FEATURES)

    explainer = LimeTabularExplainer(
        X_train[:500],
        feature_names=all_features,
        class_names=["Negative", "Positive"],
        mode="classification",
        discretize_continuous=True
    )

    os.makedirs("outputs/lime", exist_ok=True)

    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

    for idx in sample_indices:
        print(f"  LIME explanation for sample {idx}...")
        instance = X_test[idx]

        for i, disease in enumerate(DISEASE_LABELS):
            estimator = model.estimators_[i]

            def predict_fn(X):
                return np.column_stack([1 - estimator.predict_proba(X)[:, 1],
                                       estimator.predict_proba(X)[:, 1]])

            try:
                exp = explainer.explain_instance(
                    instance, predict_fn, num_features=10
                )
                html = exp.as_html()
                with open(f"outputs/lime/lime_sample{idx}_{disease}.html", "w") as f:
                    f.write(html)
            except Exception as e:
                pass

    print("LIME analysis complete. Results saved to outputs/lime/")


if __name__ == "__main__":
    run_shap_analysis()
    run_lime_analysis()
