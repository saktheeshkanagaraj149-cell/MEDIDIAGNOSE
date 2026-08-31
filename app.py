import os
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model = None
preprocessor = None
DISEASE_LABELS = [
    "Diabetes", "Heart_Disease", "Thyroid", "Hypertension",
    "Chronic_Kidney_Disease", "Liver_Disease", "Anemia",
    "Asthma", "Pneumonia", "Arthritis", "Depression", "Allergies"
]

DISEASE_DESCRIPTIONS = {
    "Diabetes": "Metabolic disorder affecting blood sugar regulation",
    "Heart_Disease": "Cardiovascular condition affecting the heart",
    "Thyroid": "Endocrine disorder of the thyroid gland",
    "Hypertension": "Persistently elevated arterial blood pressure",
    "Chronic_Kidney_Disease": "Progressive loss of kidney function",
    "Liver_Disease": "Conditions affecting liver function",
    "Anemia": "Deficiency in red blood cells or hemoglobin",
    "Asthma": "Chronic respiratory condition affecting airways",
    "Pneumonia": "Infection causing inflammation of lung air sacs",
    "Arthritis": "Inflammation of one or more joints",
    "Depression": "Mood disorder causing persistent feelings of sadness",
    "Allergies": "Immune system hypersensitivity reactions"
}

DISEASE_RISK_FACTORS = {
    "Diabetes": ["blood_glucose", "bmi", "age", "frequent_urination", "excessive_thirst"],
    "Heart_Disease": ["systolic_bp", "cholesterol_total", "age", "smoking_status", "bmi"],
    "Thyroid": ["tsh_level", "weight_gain", "weight_loss", "fatigue"],
    "Hypertension": ["systolic_bp", "diastolic_bp", "age", "bmi", "sodium_intake"],
    "Chronic_Kidney_Disease": ["creatinine", "blood_urea", "age", "systolic_bp"],
    "Liver_Disease": ["alt_level", "ast_level", "bmi", "alcohol_use"],
    "Anemia": ["hemoglobin", "fatigue", "dizziness", "gender"],
    "Asthma": ["oxygen_saturation", "respiratory_rate", "cough", "smoking_status"],
    "Pneumonia": ["wbc_count", "fever", "cough", "oxygen_saturation"],
    "Arthritis": ["joint_pain", "stiffness", "age", "bmi"],
    "Depression": ["mood_swings", "insomnia", "appetite_loss", "anxiety"],
    "Allergies": ["skin_rash", "itching", "sore_throat", "nasal_congestion"]
}


def load_model():
    global model, preprocessor
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "best_model.pkl")
    preprocessor_path = os.path.join(base_dir, "models", "preprocessor.pkl")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)


@app.route("/")
def index():
    return render_template("index.html", diseases=DISEASE_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    global model, preprocessor

    if model is None:
        load_model()

    try:
        data = request.get_json()

        features = {}

        numerical_values = {
            "age": float(data.get("age", 40)),
            "bmi": float(data.get("bmi", 25)),
            "blood_glucose": float(data.get("blood_glucose", 100)),
            "cholesterol_total": float(data.get("cholesterol_total", 200)),
            "hdl_cholesterol": float(data.get("hdl_cholesterol", 50)),
            "ldl_cholesterol": float(data.get("ldl_cholesterol", 120)),
            "triglycerides": float(data.get("triglycerides", 150)),
            "systolic_bp": float(data.get("systolic_bp", 120)),
            "diastolic_bp": float(data.get("diastolic_bp", 80)),
            "heart_rate": float(data.get("heart_rate", 75)),
            "hemoglobin": float(data.get("hemoglobin", 14)),
            "wbc_count": float(data.get("wbc_count", 7)),
            "platelet_count": float(data.get("platelet_count", 250)),
            "creatinine": float(data.get("creatinine", 1.0)),
            "blood_urea": float(data.get("blood_urea", 15)),
            "tsh_level": float(data.get("tsh_level", 2.5)),
            "alt_level": float(data.get("alt_level", 30)),
            "ast_level": float(data.get("ast_level", 28)),
            "oxygen_saturation": float(data.get("oxygen_saturation", 97)),
            "respiratory_rate": float(data.get("respiratory_rate", 16))
        }

        symptoms = [
            "fatigue", "weight_loss", "weight_gain", "frequent_urination",
            "excessive_thirst", "blurred_vision", "chest_pain",
            "shortness_of_breath", "palpitations", "swelling_legs",
            "joint_pain", "stiffness", "cough", "fever",
            "night_sweats", "mood_swings", "insomnia", "appetite_loss",
            "headache", "dizziness", "numbness", "skin_rash",
            "itching", "dry_skin", "hair_loss", "sore_throat",
            "nausea", "vomiting", "abdominal_pain", "diarrhea",
            "constipation", "frequent_infections", "easy_bruising",
            "swollen_lymph_nodes", "anxiety"
        ]

        for sym in symptoms:
            features[sym] = int(data.get(sym, 0))

        features["gender"] = data.get("gender", "Male")
        features["smoking_status"] = data.get("smoking_status", "never")

        features.update(numerical_values)

        import pandas as pd
        X = pd.DataFrame([features])

        NUMERICAL_FEATURES = [
            "age", "bmi", "blood_glucose", "cholesterol_total",
            "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
            "systolic_bp", "diastolic_bp", "heart_rate",
            "hemoglobin", "wbc_count", "platelet_count",
            "creatinine", "blood_urea", "tsh_level",
            "alt_level", "ast_level", "oxygen_saturation",
            "respiratory_rate"
        ]
        CATEGORICAL_FEATURES = ["gender", "smoking_status"]

        X = X[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + symptoms]

        X_processed = preprocessor.transform(X)

        prediction = model.predict(X_processed)
        proba = model.predict_proba(X_processed)

        results = []
        for i, disease in enumerate(DISEASE_LABELS):
            if isinstance(proba, list):
                p = proba[i][0][1] if len(proba[i].shape) > 1 else proba[i][0]
            else:
                p = proba[0][i] if proba.shape[1] > i else 0

            risk_level = "Low"
            risk_color = "#4caf50"
            if p > 0.7:
                risk_level = "High"
                risk_color = "#f44336"
            elif p > 0.4:
                risk_level = "Medium"
                risk_color = "#ff9800"

            risk_factors = []
            for factor in DISEASE_RISK_FACTORS.get(disease, []):
                if factor in numerical_values:
                    val = numerical_values[factor]
                    risk_factors.append({"factor": factor, "value": val})

            results.append({
                "disease": disease,
                "description": DISEASE_DESCRIPTIONS.get(disease, ""),
                "probability": round(float(p) * 100, 2),
                "prediction": int(prediction[0][i]),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "risk_factors": risk_factors
            })

        results.sort(key=lambda x: x["probability"], reverse=True)

        return jsonify({
            "success": True,
            "results": results,
            "top_risk": results[0] if results else None
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/model-info")
def model_info():
    try:
        import json
        results_path = os.path.join(os.path.dirname(__file__), "outputs", "model_results.json")
        with open(results_path) as f:
            results = json.load(f)

        comparison = []
        for model_name, data in results.items():
            comparison.append({
                "name": model_name.replace("_", " "),
                "accuracy": data["averages"]["avg_accuracy"],
                "precision": data["averages"]["avg_precision"],
                "recall": data["averages"]["avg_recall"],
                "f1": data["averages"]["avg_f1_score"],
                "roc_auc": data["averages"]["avg_roc_auc"]
            })

        return jsonify({"success": True, "models": comparison})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
