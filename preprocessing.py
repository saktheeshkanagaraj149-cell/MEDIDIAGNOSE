import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

SYMPTOM_FEATURES = [
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

DISEASE_LABELS = [
    "Diabetes", "Heart_Disease", "Thyroid", "Hypertension",
    "Chronic_Kidney_Disease", "Liver_Disease", "Anemia",
    "Asthma", "Pneumonia", "Arthritis", "Depression", "Allergies"
]


def build_preprocessor():
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("sym", "passthrough", SYMPTOM_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def preprocess_data(csv_path="data/medical_data.csv", test_size=0.2):
    df = pd.read_csv(csv_path)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + SYMPTOM_FEATURES]
    y = df[DISEASE_LABELS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=None
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERICAL_FEATURES + cat_feature_names + SYMPTOM_FEATURES

    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    return X_train_processed, X_test_processed, y_train, y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = preprocess_data()
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Labels: {DISEASE_LABELS}")
