# ============================================
# SHAP EXPLAINABILITY
# Credit Risk Prediction System
# ============================================

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ============================================
# 1. LOAD MODEL AND PREPROCESSOR
# ============================================

model = joblib.load("model/logistic_regression_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

print("Model and preprocessor loaded successfully.")


# ============================================
# 2. LOAD DATASET
# ============================================

data_path = "data/german.data"

columns = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "credit_risk"
]

df = pd.read_csv(
    data_path,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset loaded successfully.")


# ============================================
# 3. SEPARATE INPUT AND TARGET
# ============================================

X = df.drop("credit_risk", axis=1)

y = df["credit_risk"].map({
    1: 0,   # Good credit
    2: 1    # Bad credit
})


# ============================================
# 4. PREPROCESS DATA
# ============================================

X_processed = preprocessor.transform(X)

print("Data preprocessing completed.")


# ============================================
# 5. GET FEATURE NAMES
# ============================================

feature_names = preprocessor.get_feature_names_out()

print("Number of transformed features:", len(feature_names))


# ============================================
# 6. CREATE SHAP EXPLAINER
# ============================================

explainer = shap.LinearExplainer(
    model,
    X_processed
)

shap_values = explainer.shap_values(X_processed)

print("SHAP values calculated successfully.")


# ============================================
# 7. CREATE EXPLAINABILITY FOLDER
# ============================================

os.makedirs("explainability/results", exist_ok=True)


# ============================================
# 8. GLOBAL FEATURE IMPORTANCE
# ============================================

mean_abs_shap = abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Importance": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="SHAP Importance",
    ascending=False
)


# ============================================
# 9. DISPLAY TOP FEATURES
# ============================================

print("\n============================================")
print("TOP 15 IMPORTANT FEATURES")
print("============================================")

print(
    importance_df.head(15).to_string(index=False)
)


# ============================================
# 10. SAVE FEATURE IMPORTANCE
# ============================================

importance_df.to_csv(
    "explainability/results/shap_feature_importance.csv",
    index=False
)

print("\nSHAP feature importance saved.")


# ============================================
# 11. CREATE SHAP SUMMARY PLOT
# ============================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_processed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "explainability/results/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP summary plot saved.")


# ============================================
# COMPLETE
# ============================================

print("\n============================================")
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("============================================")