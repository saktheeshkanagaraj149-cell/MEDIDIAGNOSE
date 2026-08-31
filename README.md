# 🩺 MediDiagnose: Multi-Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-brightgreen?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red?logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **AI-powered multi-disease prediction system** that predicts the risk of **12 diseases** from patient symptoms, lab test results, and demographic data using ensemble machine learning models — all wrapped in a modern, interactive Flask web application.

> ⚠️ **Disclaimer:** This project is for **educational and research purposes only**. It is **not** a substitute for professional medical diagnosis, advice, or treatment. Always consult qualified healthcare providers for medical decisions.

---

## 🌟 Features

- **12 Disease Predictions** — Diabetes, Heart Disease, Thyroid, Hypertension, Chronic Kidney Disease, Liver Disease, Anemia, Asthma, Pneumonia, Arthritis, Depression & Allergies
- **Multi-Label Classification** — Predicts multiple diseases simultaneously (comorbidities included)
- **3 ML Models Compared** — Logistic Regression, Random Forest & XGBoost
- **Interactive Web UI** — Dark-themed dashboard with live Chart.js visualizations
- **Interpretability** — SHAP & LIME explanations for model transparency
- **Automated EDA Report** — Rich HTML report with 10+ visualizations
- **Fully Reproducible** — Seeded generation, complete pipeline automation

---

## 🧠 Diseases Predicted

| # | Disease | Key Indicators |
|---|---------|---------------|
| 1 | Diabetes | Blood glucose, BMI, frequent urination |
| 2 | Heart Disease | BP, cholesterol, smoking, age |
| 3 | Thyroid | TSH levels, weight changes, fatigue |
| 4 | Hypertension | Systolic/diastolic BP, age, BMI |
| 5 | Chronic Kidney Disease | Creatinine, blood urea, BP |
| 6 | Liver Disease | ALT/AST levels, BMI |
| 7 | Anemia | Hemoglobin, fatigue, dizziness |
| 8 | Asthma | O₂ saturation, respiratory rate, cough |
| 9 | Pneumonia | WBC count, fever, oxygen levels |
| 10 | Arthritis | Joint pain, stiffness, age |
| 11 | Depression | Mood swings, insomnia, appetite loss |
| 12 | Allergies | Skin rash, itching, sore throat |

---

## 📊 Project Structure

```
MediDiagnose/
├── app.py                  # Flask web application (predict + model-info APIs)
├── run_pipeline.py         # One-command full pipeline runner
├── generate_data.py        # Synthetic medical dataset generator
├── preprocessing.py        # Imputation, OneHotEncoder, StandardScaler, split
├── train.py                # Train & evaluate LR, RF, XGBoost
├── generate_eda.py         # EDA HTML report generator
├── interpretability.py     # SHAP & LIME analysis
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── data/                   # Generated datasets (gitignored)
│   ├── .gitkeep
│   └── medical_data.csv    # 5000 patients × 69 columns
├── models/                 # Saved models (gitignored)
│   ├── .gitkeep
│   ├── best_model.pkl      # XGBoost (selected by avg F1)
│   ├── preprocessor.pkl    # Fitted ColumnTransformer
│   ├── feature_names.pkl
│   └── disease_labels.pkl
├── outputs/                # Results & reports (gitignored)
│   ├── eda_report.html
│   ├── model_results.json
│   ├── model_comparison.csv
│   └── shap/               # SHAP visualizations
├── templates/
│   └── index.html          # Frontend dashboard
└── static/
    └── style.css           # Styling
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **pip** package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/saktheeshkanagaraj149-cell/MEDIDIAGNOSE.git
cd MEDIDIAGNOSE

# Create & activate virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python run_pipeline.py
```

This performs, in sequence:
1. **Data Generation** → 5,000 synthetic patient records
2. **EDA Report** → `outputs/eda_report.html`
3. **Model Training** → Logistic Regression, Random Forest, XGBoost
4. **Evaluation** → 6 metrics per disease + confusion matrices
5. **Interpretability** → SHAP analysis for all 12 diseases

### Launch the Web App

```bash
python app.py
```

Then open **[http://localhost:5000](http://localhost:5000)** in your browser.

> **Note:** Run the pipeline at least once before starting the app so the model files exist.

---

## 🖥️ Web Application

The web UI features:
- **Input forms** for demographics, vitals, blood tests & 35 symptom checkboxes
- **8 Chart.js visualizations**:
  - Horizontal bar chart (all disease probabilities)
  - Doughnut chart (risk distribution)
  - Radar chart (top-5 risk profile)
  - Polar area chart (probability vs threshold)
  - Grouped bar chart (model comparison)
- **Ranked results** with risk badges & probability bars
- **Model comparison** page

---

## 🧪 Model Performance

Models were trained on 80/20 split of 5,000 patient records and evaluated across all 12 diseases.

| Model | Avg Accuracy | Avg Precision | Avg Recall | Avg F1 | Avg ROC-AUC |
|-------|:------------:|:-------------:|:----------:|:------:|:-----------:|
| Logistic Regression | 90.8% | 32.6% | 16.8% | 18.7% | 49.0% |
| Random Forest | 91.2% | 30.8% | 17.8% | 19.7% | 47.4% |
| **XGBoost** ✅ | **91.5%** | **48.9%** | **25.0%** | **27.5%** | **48.5%** |

> **XGBoost** is selected as the best model (highest average F1-score) and saved as `best_model.pkl`.

Per-disease performance is strong on high-prevalence conditions:
- **Diabetes:** F1 = 87.8%, **Hypertension:** F1 = 95.1%

---

## 🔬 Technical Details

### Data Preprocessing
- **Missing Values:** Median imputation (numerical), most-frequent (categorical)
- **Encoding:** `OneHotEncoder` with `drop='first'` to avoid multicollinearity
- **Scaling:** `StandardScaler` (zero mean, unit variance)
- **Split:** 80/20 train/test

### Models (MultiOutputClassifier)
| Model | Type | Key Parameters |
|-------|------|---------------|
| Logistic Regression | Linear | max_iter=2000, C=1.0 |
| Random Forest | Ensemble | n_estimators=200, max_depth=15 |
| XGBoost | Gradient Boosting | n_estimators=300, max_depth=8, lr=0.08 |

### Evaluation Metrics
Per disease: **Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix**

### Interpretability
- **SHAP** — TreeExplainer for global feature importance & interaction effects
- **LIME** — Local interpretable model-agnostic explanations per prediction

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-------|--------------|
| Backend | Python, Flask |
| Machine Learning | Scikit-learn, XGBoost |
| Interpretability | SHAP, LIME |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Frontend | HTML5, CSS3, Vanilla JavaScript |

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 👤 Author

**Saktheesh Kanagaraj**

---

## 📬 Contact

- **GitHub:** [saktheeshkanagaraj149-cell](https://github.com/saktheeshkanagaraj149-cell)
- **Repository:** [MEDIDIAGNOSE](https://github.com/saktheeshkanagaraj149-cell/MEDIDIAGNOSE)
