# ============================================
# LIME EXPLAINABILITY
# Credit Risk Prediction System
# ============================================

import pandas as pd
import joblib
import os

from lime.lime_tabular import LimeTabularExplainer


# ============================================
# 1. LOAD MODEL AND PREPROCESSOR
# ============================================

model = joblib.load("model/logistic_regression_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

print("Model and preprocessor loaded successfully.")


# ============================================
# 2. LOAD DATASET
# ============================================

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
    "data/german.data",
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset loaded successfully.")


# ============================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================

X = df.drop("credit_risk", axis=1)

y = df["credit_risk"].map({
    1: 0,
    2: 1
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


# ============================================
# 6. CREATE LIME PREDICTION FUNCTION
# ============================================

def predict_function(data):
    """
    LIME provides transformed numerical data.
    The Logistic Regression model directly
    receives this transformed data.
    """

    return model.predict_proba(data)


# ============================================
# 7. CREATE LIME EXPLAINER
# ============================================

explainer = LimeTabularExplainer(
    X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
    feature_names=feature_names,
    class_names=["Low Risk", "High Risk"],
    mode="classification",
    random_state=42
)

print("LIME explainer created.")


# ============================================
# 8. SELECT ONE APPLICANT
# ============================================

applicant_index = 0

applicant = X_processed[applicant_index]

if hasattr(applicant, "toarray"):
    applicant = applicant.toarray().flatten()
else:
    applicant = applicant.flatten()


# ============================================
# 9. GENERATE LIME EXPLANATION
# ============================================

explanation = explainer.explain_instance(
    applicant,
    predict_function,
    num_features=15
)


# ============================================
# 10. DISPLAY PREDICTION
# ============================================

prediction = model.predict_proba(
    applicant.reshape(1, -1)
)[0]

print("\n============================================")
print("LIME APPLICANT PREDICTION")
print("============================================")

print("Low Risk Probability :", round(prediction[0] * 100, 2), "%")
print("High Risk Probability:", round(prediction[1] * 100, 2), "%")


# ============================================
# 11. DISPLAY LIME EXPLANATION
# ============================================

print("\n============================================")
print("LIME EXPLANATION")
print("============================================")

for feature, weight in explanation.as_list():
    print(f"{feature}: {weight:.4f}")


# ============================================
# 12. CREATE RESULTS FOLDER
# ============================================

os.makedirs("explainability/results", exist_ok=True)


# ============================================
# 13. SAVE LIME EXPLANATION
# ============================================

explanation.save_to_file(
    "explainability/results/lime_explanation.html"
)

print("\nLIME explanation saved.")


# ============================================
# COMPLETE
# ============================================

print("\n============================================")
print("LIME ANALYSIS COMPLETED SUCCESSFULLY")
print("============================================")