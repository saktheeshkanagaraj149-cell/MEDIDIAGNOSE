"""
MediDiagnose: Multi-Disease Prediction System
Main pipeline runner - executes all steps in sequence.
"""
import sys
import os

def main():
    print("=" * 70)
    print("  MediDiagnose: Multi-Disease Prediction System")
    print("  Pipeline Runner")
    print("=" * 70)

    print("\n[1/5] Generating synthetic medical dataset...")
    from generate_data import generate_dataset
    df = generate_dataset(5000)
    df.to_csv("data/medical_data.csv", index=False)
    print(f"  -> Dataset: {df.shape[0]} samples, {df.shape[1]} columns saved to data/medical_data.csv")

    print("\n[2/5] Generating EDA report...")
    from generate_eda import generate_eda_report
    generate_eda_report("data/medical_data.csv", "outputs/eda_report.html")
    print("  -> EDA report saved to outputs/eda_report.html")

    print("\n[3/5] Training and evaluating models...")
    from train import train_and_evaluate
    results, trained_models = train_and_evaluate()
    print("  -> Models trained and saved to models/")

    print("\n[4/5] Running interpretability analysis (SHAP)...")
    try:
        from interpretability import run_shap_analysis
        run_shap_analysis()
    except Exception as e:
        print(f"  -> SHAP analysis skipped: {e}")
    print("  -> Interpretability results saved to outputs/shap/")

    print("\n[5/5] All steps complete!")
    print("=" * 70)
    print("  Outputs:")
    print("    - data/medical_data.csv          (Dataset)")
    print("    - outputs/eda_report.html        (EDA Visualizations)")
    print("    - models/best_model.pkl          (Trained Model)")
    print("    - models/preprocessor.pkl        (Preprocessing Pipeline)")
    print("    - outputs/model_results.json     (Detailed Metrics)")
    print("    - outputs/model_comparison.csv   (Model Comparison)")
    print("    - outputs/shap/                  (SHAP Analysis)")
    print("=" * 70)
    print("\n  To run the Flask app: python app.py")
    print("  Then open http://localhost:5000")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
