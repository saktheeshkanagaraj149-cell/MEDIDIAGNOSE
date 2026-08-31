import numpy as np
import pandas as pd

DISEASES = [
    "Diabetes", "Heart_Disease", "Thyroid", "Hypertension",
    "Chronic_Kidney_Disease", "Liver_Disease", "Anemia",
    "Asthma", "Pneumonia", "Arthritis", "Depression", "Allergies"
]

SYMPTOM_COLUMNS = [
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

NUMERICAL_COLUMNS = [
    "age", "bmi", "blood_glucose", "cholesterol_total",
    "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
    "systolic_bp", "diastolic_bp", "heart_rate",
    "hemoglobin", "wbc_count", "platelet_count",
    "creatinine", "blood_urea", "tsh_level",
    "alt_level", "ast_level", "oxygen_saturation",
    "respiratory_rate"
]

CATEGORICAL_COLUMNS = ["gender", "smoking_status"]

SEED = 42


def _clip(val, lo, hi):
    return max(lo, min(hi, val))


def _generate_base_features(n):
    rng = np.random.RandomState(SEED)
    df = pd.DataFrame()

    df["age"] = rng.normal(45, 18, n).clip(1, 95).round(0).astype(int)
    df["gender"] = rng.choice(["Male", "Female"], n)
    df["bmi"] = rng.normal(27, 5, n).clip(14, 60).round(1)
    df["smoking_status"] = rng.choice(
        ["never", "former", "current"], n, p=[0.5, 0.3, 0.2]
    )

    df["blood_glucose"] = rng.normal(110, 35, n).clip(60, 400).round(0).astype(int)
    df["cholesterol_total"] = rng.normal(200, 40, n).clip(100, 400).round(0).astype(int)
    df["hdl_cholesterol"] = rng.normal(50, 15, n).clip(15, 120).round(0).astype(int)
    df["ldl_cholesterol"] = rng.normal(120, 35, n).clip(40, 300).round(0).astype(int)
    df["triglycerides"] = rng.normal(150, 60, n).clip(40, 600).round(0).astype(int)
    df["systolic_bp"] = rng.normal(125, 18, n).clip(80, 220).round(0).astype(int)
    df["diastolic_bp"] = rng.normal(80, 12, n).clip(40, 140).round(0).astype(int)
    df["heart_rate"] = rng.normal(75, 12, n).clip(45, 150).round(0).astype(int)
    df["hemoglobin"] = rng.normal(14, 2, n).clip(6, 20).round(1)
    df["wbc_count"] = rng.normal(7.5, 2.5, n).clip(1, 20).round(1)
    df["platelet_count"] = rng.normal(250, 80, n).clip(50, 600).round(0).astype(int)
    df["creatinine"] = rng.normal(1.0, 0.4, n).clip(0.3, 8).round(2)
    df["blood_urea"] = rng.normal(15, 6, n).clip(5, 80).round(1)
    df["tsh_level"] = rng.normal(2.5, 1.5, n).clip(0.1, 15).round(2)
    df["alt_level"] = rng.normal(30, 15, n).clip(5, 200).round(0).astype(int)
    df["ast_level"] = rng.normal(28, 14, n).clip(5, 200).round(0).astype(int)
    df["oxygen_saturation"] = rng.normal(97, 3, n).clip(70, 100).round(0).astype(int)
    df["respiratory_rate"] = rng.normal(16, 3, n).clip(10, 40).round(0).astype(int)

    symptoms = rng.binomial(1, 0.12, size=(n, len(SYMPTOM_COLUMNS)))
    for i, col in enumerate(SYMPTOM_COLUMNS):
        df[col] = symptoms[:, i].astype(int)

    return df


def _apply_disease_rules(df, rng):
    n = len(df)
    labels = pd.DataFrame(0, index=range(n), columns=DISEASES)

    age = df["age"].values
    bmi = df["bmi"].values
    glucose = df["blood_glucose"].values
    chol = df["cholesterol_total"].values
    hdl = df["hdl_cholesterol"].values
    ldl = df["ldl_cholesterol"].values
    trig = df["triglycerides"].values
    sbp = df["systolic_bp"].values
    dbp = df["diastolic_bp"].values
    hr = df["heart_rate"].values
    hgb = df["hemoglobin"].values
    wbc = df["wbc_count"].values
    plt_count = df["platelet_count"].values
    cr = df["creatinine"].values
    bun = df["blood_urea"].values
    tsh = df["tsh_level"].values
    alt = df["alt_level"].values
    ast = df["ast_level"].values
    o2 = df["oxygen_saturation"].values
    resp = df["respiratory_rate"].values

    smoking = df["smoking_status"].values
    gender = df["gender"].values

    def add_noise(prob):
        noise = rng.normal(0, 0.04, len(prob))
        return np.clip(prob + noise, 0, 1)

    p_diabetes = np.where(
        (glucose > 126) & (age > 35), 0.85,
        np.where((glucose > 110) & (bmi > 30), 0.55,
        np.where((glucose > 100) & (age > 45), 0.35, 0.05))
    )
    p_diabetes = add_noise(p_diabetes)
    labels["Diabetes"] = rng.binomial(1, p_diabetes)
    df.loc[labels["Diabetes"] == 1, "blood_glucose"] = rng.randint(126, 350, labels["Diabetes"].sum())

    p_heart = np.where(
        (age > 55) & (sbp > 140) & (chol > 240), 0.8,
        np.where((age > 50) & (smoking == "current") & (ldl > 160), 0.65,
        np.where((sbp > 130) & (bmi > 30) & (hr > 85), 0.45, 0.06))
    )
    p_heart = add_noise(p_heart)
    labels["Heart_Disease"] = rng.binomial(1, p_heart)

    p_hyper = np.where(
        (sbp > 140) | (dbp > 90), 0.85,
        np.where((sbp > 130) & (dbp > 85), 0.5,
        np.where((age > 50) & (bmi > 30), 0.3, 0.04))
    )
    p_hyper = add_noise(p_hyper)
    labels["Hypertension"] = rng.binomial(1, p_hyper)
    df.loc[labels["Hypertension"] == 1, "systolic_bp"] = rng.randint(140, 210, labels["Hypertension"].sum())
    df.loc[labels["Hypertension"] == 1, "diastolic_bp"] = rng.randint(90, 130, labels["Hypertension"].sum())

    p_thyroid = np.where(
        (tsh > 5.0) | (tsh < 0.3), 0.8,
        np.where((tsh > 3.5) | (tsh < 0.5), 0.45, 0.04)
    )
    p_thyroid = add_noise(p_thyroid)
    labels["Thyroid"] = rng.binomial(1, p_thyroid)

    p_ckd = np.where(
        (cr > 2.0) | (bun > 40), 0.8,
        np.where((cr > 1.5) | (bun > 25), 0.45,
        np.where((age > 60) & (sbp > 150), 0.3, 0.03))
    )
    p_ckd = add_noise(p_ckd)
    labels["Chronic_Kidney_Disease"] = rng.binomial(1, p_ckd)

    p_liver = np.where(
        (alt > 80) | (ast > 80), 0.75,
        np.where((alt > 50) & (bmi > 32), 0.4,
        np.where((alt > 40) & (smoking == "current"), 0.25, 0.04))
    )
    p_liver = add_noise(p_liver)
    labels["Liver_Disease"] = rng.binomial(1, p_liver)

    p_anemia = np.where(
        (hgb < 10), 0.9,
        np.where((hgb < 12) & (gender == "Female"), 0.5,
        np.where((hgb < 12), 0.35, 0.05))
    )
    p_anemia = add_noise(p_anemia)
    labels["Anemia"] = rng.binomial(1, p_anemia)

    p_asthma = np.where(
        (o2 < 93) & (resp > 22), 0.7,
        np.where((o2 < 95) & (df["cough"] == 1), 0.5,
        np.where((smoking == "current") & (resp > 18), 0.3, 0.06))
    )
    p_asthma = add_noise(p_asthma)
    labels["Asthma"] = rng.binomial(1, p_asthma)

    p_pneumonia = np.where(
        (wbc > 12) & (o2 < 92) & (df["fever"] == 1), 0.85,
        np.where((wbc > 10) & (df["cough"] == 1) & (resp > 20), 0.5,
        np.where((wbc > 9) & (df["fever"] == 1) & (df["cough"] == 1), 0.35, 0.03))
    )
    p_pneumonia = add_noise(p_pneumonia)
    labels["Pneumonia"] = rng.binomial(1, p_pneumonia)

    p_arthritis = np.where(
        (age > 50) & (df["joint_pain"] == 1) & (df["stiffness"] == 1), 0.7,
        np.where((age > 40) & (df["joint_pain"] == 1), 0.4,
        np.where((bmi > 30) & (age > 35), 0.25, 0.05))
    )
    p_arthritis = add_noise(p_arthritis)
    labels["Arthritis"] = rng.binomial(1, p_arthritis)

    p_depression = np.where(
        (df["mood_swings"] == 1) & (df["insomnia"] == 1) & (df["appetite_loss"] == 1), 0.75,
        np.where((df["mood_swings"] == 1) & (df["insomnia"] == 1), 0.5,
        np.where((df["anxiety"] == 1) & (df["fatigue"] == 1), 0.35, 0.08))
    )
    p_depression = add_noise(p_depression)
    labels["Depression"] = rng.binomial(1, p_depression)

    p_allergy = np.where(
        (df["skin_rash"] == 1) & (df["itching"] == 1) & (df["sore_throat"] == 1), 0.7,
        np.where((df["skin_rash"] == 1) & (df["itching"] == 1), 0.5,
        np.where((df["sneezing"] == 1) if "sneezing" in df.columns else False, 0.3, 0.08))
    )
    p_allergy = add_noise(p_allergy)
    labels["Allergies"] = rng.binomial(1, p_allergy)

    for i in range(n):
        row_diseases = labels.columns[labels.iloc[i] == 1]
        if len(row_diseases) > 4:
            keep = rng.choice(row_diseases, size=min(3, len(row_diseases)), replace=False)
            for d in row_diseases:
                if d not in keep:
                    labels.at[i, d] = 0

    return labels


def generate_dataset(n_samples=5000):
    rng = np.random.RandomState(SEED)
    df = _generate_base_features(n_samples)
    labels = _apply_disease_rules(df, rng)

    df = pd.concat([df.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)

    missing_mask = rng.random(df[NUMERICAL_COLUMNS].shape) < 0.03
    for i, col in enumerate(NUMERICAL_COLUMNS):
        df.loc[missing_mask[:, i], col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_dataset(5000)
    df.to_csv("data/medical_data.csv", index=False)
    print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Diseases: {DISEASES}")
    print(f"\nDisease prevalence:")
    for d in DISEASES:
        print(f"  {d}: {df[d].mean():.1%}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
