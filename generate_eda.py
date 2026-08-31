import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
from preprocessing import NUMERICAL_FEATURES, DISEASE_LABELS


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def generate_eda_report(csv_path="data/medical_data.csv", output_path="outputs/eda_report.html"):
    df = pd.read_csv(csv_path)

    sns.set_theme(style="whitegrid", palette="muted")
    colors = sns.color_palette("Set2", 12)

    sections = []

    sections.append("""
    <div class="section">
        <h2>1. Dataset Overview</h2>
        <table class="stats-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Samples</td><td>{}</td></tr>
            <tr><td>Total Features</td><td>{}</td></tr>
            <tr><td>Numerical Features</td><td>{}</td></tr>
            <tr><td>Binary Symptom Features</td><td>{}</td></tr>
            <tr><td>Categorical Features</td><td>2</td></tr>
            <tr><td>Disease Labels</td><td>{}</td></tr>
            <tr><td>Missing Values</td><td>{}</td></tr>
        </table>
    </div>
    """.format(
        len(df), len(df.columns),
        len(NUMERICAL_FEATURES),
        len(df.columns) - len(NUMERICAL_FEATURES) - 2 - len(DISEASE_LABELS),
        len(DISEASE_LABELS),
        df.isnull().sum().sum()
    ))

    fig, ax = plt.subplots(figsize=(12, 5))
    disease_prevalence = df[DISEASE_LABELS].mean().sort_values(ascending=False)
    disease_prevalence.plot(kind="bar", ax=ax, color=colors[:len(DISEASE_LABELS)])
    ax.set_title("Disease Prevalence", fontsize=14, fontweight="bold")
    ax.set_ylabel("Prevalence")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(disease_prevalence.values):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=8)
    plt.tight_layout()
    sections.append(f"""
    <div class="section">
        <h2>2. Disease Prevalence</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" class="plot">
    </div>
    """)

    fig, axes = plt.subplots(4, 5, figsize=(22, 16))
    for i, col in enumerate(NUMERICAL_FEATURES):
        row, c = divmod(i, 5)
        ax = axes[row][c]
        df[col].hist(ax=ax, bins=30, color=colors[i % len(colors)], edgecolor="white", alpha=0.8)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
    for j in range(i + 1, 20):
        row, c = divmod(j, 5)
        axes[row][c].set_visible(False)
    plt.suptitle("Numerical Feature Distributions", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    sections.append(f"""
    <div class="section">
        <h2>3. Numerical Feature Distributions</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" class="plot">
    </div>
    """)

    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df[NUMERICAL_FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, annot_kws={"size": 7},
                square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    sections.append(f"""
    <div class="section">
        <h2>4. Feature Correlation Heatmap</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" class="plot">
    </div>
    """)

    key_features = ["blood_glucose", "systolic_bp", "cholesterol_total", "hemoglobin", "bmi", "creatinine"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, feat in enumerate(key_features):
        row, c = divmod(i, 3)
        ax = axes[row][c]
        for j, disease in enumerate(DISEASE_LABELS[:6]):
            subset = df[df[disease] == 1][feat].dropna()
            if len(subset) > 0:
                ax.hist(subset, bins=25, alpha=0.4, label=disease, density=True)
        ax.set_title(f"{feat} by Disease", fontsize=10, fontweight="bold")
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    plt.suptitle("Key Features by Disease Status", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    sections.append(f"""
    <div class="section">
        <h2>5. Key Features by Disease Status</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" class="plot">
    </div>
    """)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    box_features = ["blood_glucose", "systolic_bp", "cholesterol_total", "hemoglobin", "tsh_level", "creatinine"]
    for i, feat in enumerate(box_features):
        row, c = divmod(i, 3)
        ax = axes[row][c]
        target_disease = DISEASE_LABELS[i]
        df.boxplot(column=feat, by=target_disease, ax=ax)
        ax.set_title(f"{feat} by {target_disease}", fontsize=10, fontweight="bold")
        ax.set_xlabel(target_disease)
        ax.set_ylabel(feat)
    fig.suptitle("")
    plt.suptitle("Box Plots: Feature Ranges by Disease", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    sections.append(f"""
    <div class="section">
        <h2>6. Box Plots</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" class="plot">
    </div>
    """)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    scatter_pairs = [
        ("blood_glucose", "bmi"), ("systolic_bp", "diastolic_bp"),
        ("cholesterol_total", "hdl_cholesterol"), ("creatinine", "blood_urea")
    ]
    for i, (x_feat, y_feat) in enumerate(scatter_pairs):
        row, c = divmod(i, 2)
        ax = axes[row][c]
        for j, disease in enumerate(DISEASE_LABELS[:4]):
            subset = df[df[disease] == 1]
            ax.scatter(subset[x_feat], subset[y_feat], alpha=0.3, s=10, label=disease)
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.set_title(f"{x_feat} vs {y_feat}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
    plt.suptitle("Feature Scatter Plots", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    sections.append(f"""
    <div class="section">
        <h2>7. Scatter Plots</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" class="plot">
    </div>
    """)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, disease in enumerate(DISEASE_LABELS[:4]):
        row, c = divmod(i, 2)
        ax = axes[row][c]
        cooccur = df[DISEASE_LABELS].corr()[disease].drop(disease).sort_values(ascending=False)
        cooccur.plot(kind="barh", ax=ax, color=[colors[0] if v > 0 else colors[1] for v in cooccur.values])
        ax.set_title(f"Co-occurrence with {disease}", fontsize=10, fontweight="bold")
        ax.set_xlim(-1, 1)
    plt.suptitle("Disease Co-occurrence", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    sections.append(f"""
    <div class="section">
        <h2>8. Disease Co-occurrence</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" class="plot">
    </div>
    """)

    age_bins = [0, 18, 30, 45, 60, 75, 100]
    age_labels = ["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
    fig, ax = plt.subplots(figsize=(12, 6))
    age_disease = df.groupby("age_group")[DISEASE_LABELS].mean()
    age_disease.plot(kind="bar", ax=ax, colormap="tab20", width=0.8)
    ax.set_title("Disease Prevalence by Age Group", fontsize=14, fontweight="bold")
    ax.set_ylabel("Prevalence")
    ax.set_xlabel("Age Group")
    plt.xticks(rotation=0)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    sections.append(f"""
    <div class="section">
        <h2>9. Disease by Age Group</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" class="plot">
    </div>
    """)

    gender_disease = df.groupby("gender")[DISEASE_LABELS].mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    gender_disease.T.plot(kind="bar", ax=ax, colormap="Set2", width=0.7)
    ax.set_title("Disease Prevalence by Gender", fontsize=14, fontweight="bold")
    ax.set_ylabel("Prevalence")
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="Gender")
    plt.tight_layout()
    sections.append(f"""
    <div class="section">
        <h2>10. Disease by Gender</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" class="plot">
    </div>
    """)

    missing = df[NUMERICAL_FEATURES].isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        missing.sort_values().plot(kind="barh", ax=ax, color="salmon")
        ax.set_title("Missing Values by Feature", fontsize=14, fontweight="bold")
        ax.set_xlabel("Count")
        plt.tight_layout()
        sections.append(f"""
        <div class="section">
            <h2>11. Missing Values</h2>
            <img src="data:image/png;base64,{fig_to_base64(fig)}" class="plot">
        </div>
        """)

    stats_df = df[NUMERICAL_FEATURES].describe().T
    stats_html = stats_df.round(2).to_html(classes="stats-table")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MediDiagnose - EDA Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
               background: #0a0e27; color: #e0e0e0; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; padding: 40px 0;
                   background: linear-gradient(135deg, #1a1a4e, #0d3b66);
                   border-radius: 12px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2.2em; color: #00d4ff; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; color: #b0b0b0; }}
        .section {{ background: #111638; border-radius: 12px; padding: 25px;
                    margin-bottom: 25px; border: 1px solid #1e2a5e; }}
        .section h2 {{ color: #00d4ff; margin-bottom: 15px; font-size: 1.4em;
                       border-bottom: 2px solid #1e2a5e; padding-bottom: 8px; }}
        .plot {{ width: 100%; border-radius: 8px; margin: 10px 0; }}
        .stats-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        .stats-table th, .stats-table td {{ padding: 10px 14px; text-align: left;
                     border-bottom: 1px solid #1e2a5e; }}
        .stats-table th {{ background: #0d1542; color: #00d4ff; font-weight: 600; }}
        .stats-table tr:hover {{ background: #151c4a; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MediDiagnose</h1>
            <p>Exploratory Data Analysis Report</p>
            <p>Multi-Disease Prediction System &mdash; 12 Diseases</p>
        </div>
        {"".join(sections)}
        <div class="section">
            <h2>Descriptive Statistics</h2>
            {stats_html}
        </div>
        <div class="footer">
            <p>Generated by MediDiagnose EDA Pipeline</p>
        </div>
    </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"EDA report saved to {output_path}")


if __name__ == "__main__":
    generate_eda_report()
