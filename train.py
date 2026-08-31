import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
from preprocessing import preprocess_data, DISEASE_LABELS


def build_models():
    models = {
        "Logistic_Regression": MultiOutputClassifier(
            LogisticRegression(max_iter=2000, random_state=42, C=1.0),
            n_jobs=-1
        ),
        "Random_Forest": MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            n_jobs=-1
        ),
        "XGBoost": MultiOutputClassifier(
            XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.08,
                eval_metric="logloss", random_state=42, n_jobs=-1,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3
            ),
            n_jobs=-1
        )
    }
    return models


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)
        if isinstance(y_proba, list):
            y_proba = np.column_stack(y_proba)
    except Exception:
        y_proba = None

    metrics = {"model_name": model_name, "diseases": {}}

    for i, disease in enumerate(DISEASE_LABELS):
        y_true = y_test[disease].values
        y_p = y_pred[:, i]

        acc = accuracy_score(y_true, y_p)
        prec = precision_score(y_true, y_p, zero_division=0)
        rec = recall_score(y_true, y_p, zero_division=0)
        f1 = f1_score(y_true, y_p, zero_division=0)

        try:
            if y_proba is not None:
                if y_proba.ndim == 2:
                    roc = roc_auc_score(y_true, y_proba[:, i])
                else:
                    roc = 0.5
            else:
                roc = 0.5
        except Exception:
            roc = 0.5

        cm = confusion_matrix(y_true, y_p).tolist()

        metrics["diseases"][disease] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc, 4),
            "confusion_matrix": cm
        }

    avg_metrics = {}
    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        vals = [metrics["diseases"][d][metric] for d in DISEASE_LABELS]
        avg_metrics[f"avg_{metric}"] = round(np.mean(vals), 4)
    metrics["averages"] = avg_metrics

    return metrics


def train_and_evaluate():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data()

    models = build_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        print(f"Evaluating {name}...")
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics
        trained_models[name] = model
        print(f"  Avg F1: {metrics['averages']['avg_f1_score']:.4f}")
        print(f"  Avg ROC-AUC: {metrics['averages']['avg_roc_auc']:.4f}")

    best_model_name = max(
        results.keys(),
        key=lambda k: results[k]["averages"]["avg_f1_score"]
    )
    print(f"\nBest model: {best_model_name}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    joblib.dump(trained_models[best_model_name], "models/best_model.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    joblib.dump(DISEASE_LABELS, "models/disease_labels.pkl")

    with open("outputs/model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = []
    for name in results:
        summary.append({
            "Model": name,
            "Avg Accuracy": results[name]["averages"]["avg_accuracy"],
            "Avg Precision": results[name]["averages"]["avg_precision"],
            "Avg Recall": results[name]["averages"]["avg_recall"],
            "Avg F1": results[name]["averages"]["avg_f1_score"],
            "Avg ROC-AUC": results[name]["averages"]["avg_roc_auc"]
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("outputs/model_comparison.csv", index=False)

    print(f"\nResults saved to outputs/model_results.json")
    print(f"Comparison saved to outputs/model_comparison.csv")
    print(f"Best model saved to models/best_model.pkl")

    return results, trained_models


if __name__ == "__main__":
    train_and_evaluate()
