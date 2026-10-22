import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "data/german.data"

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
    file_path,
    sep=r"\s+",
    header=None,
    names=columns
)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]


# ============================================================
# 3. CONVERT TARGET
# ============================================================

# Original:
# 1 = Good credit
# 2 = Bad credit

# New:
# 0 = Good / Low Risk
# 1 = Bad / High Risk

y = y.map({
    1: 0,
    2: 1
})


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("===== CATEGORICAL FEATURES =====")
print(categorical_features)

print("\n===== NUMERICAL FEATURES =====")
print(numerical_features)


# ============================================================
# 5. CREATE PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

print("\n===== TARGET DISTRIBUTION =====")
print(y.value_counts())

print("\n===== TARGET MEANING =====")
print("0 = Good / Low Risk")
print("1 = Bad / High Risk")


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN / TEST SPLIT =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 8. PREPROCESS THE DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\n===== AFTER ENCODING =====")
print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)


# ============================================================
# 9. HANDLE CLASS IMBALANCE USING SMOTE
# ============================================================

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\n===== AFTER SMOTE =====")
print("Training samples:", len(X_train_resampled))

print("\nResampled target distribution:")
print(y_train_resampled.value_counts())
# ============================================================
# 10. LOGISTIC REGRESSION MODEL
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train the model
logistic_model.fit(
    X_train_resampled,
    y_train_resampled
)

# Make predictions
y_pred_logistic = logistic_model.predict(X_test_processed)

# Get probability of Bad / High Risk
y_prob_logistic = logistic_model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 11. EVALUATE LOGISTIC REGRESSION
# ============================================================

accuracy = accuracy_score(y_test, y_pred_logistic)

precision = precision_score(y_test, y_pred_logistic)

recall = recall_score(y_test, y_pred_logistic)

f1 = f1_score(y_test, y_pred_logistic)

auc = roc_auc_score(y_test, y_prob_logistic)


print("\n========================================")
print("       LOGISTIC REGRESSION RESULTS")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"AUC-ROC   : {auc:.4f}")
# ============================================================
# 12. RANDOM FOREST MODEL
# ============================================================

from sklearn.ensemble import RandomForestClassifier


# Create Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=None
)


# Train the model
random_forest_model.fit(
    X_train_resampled,
    y_train_resampled
)


# Make predictions
y_pred_rf = random_forest_model.predict(
    X_test_processed
)


# Get probability of Bad / High Risk
y_prob_rf = random_forest_model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 13. EVALUATE RANDOM FOREST
# ============================================================

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

rf_precision = precision_score(
    y_test,
    y_pred_rf
)

rf_recall = recall_score(
    y_test,
    y_pred_rf
)

rf_f1 = f1_score(
    y_test,
    y_pred_rf
)

rf_auc = roc_auc_score(
    y_test,
    y_prob_rf
)


print("\n========================================")
print("          RANDOM FOREST RESULTS")
print("========================================")

print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1-Score  : {rf_f1:.4f}")
print(f"AUC-ROC   : {rf_auc:.4f}")
# ============================================================
# 14. XGBOOST MODEL
# ============================================================

from xgboost import XGBClassifier


# Create XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)


# Train the model
xgb_model.fit(
    X_train_resampled,
    y_train_resampled
)


# Make predictions
y_pred_xgb = xgb_model.predict(
    X_test_processed
)


# Get probability of Bad / High Risk
y_prob_xgb = xgb_model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 15. EVALUATE XGBOOST
# ============================================================

xgb_accuracy = accuracy_score(
    y_test,
    y_pred_xgb
)

xgb_precision = precision_score(
    y_test,
    y_pred_xgb
)

xgb_recall = recall_score(
    y_test,
    y_pred_xgb
)

xgb_f1 = f1_score(
    y_test,
    y_pred_xgb
)

xgb_auc = roc_auc_score(
    y_test,
    y_prob_xgb
)


print("\n========================================")
print("            XGBOOST RESULTS")
print("========================================")

print(f"Accuracy  : {xgb_accuracy:.4f}")
print(f"Precision : {xgb_precision:.4f}")
print(f"Recall    : {xgb_recall:.4f}")
print(f"F1-Score  : {xgb_f1:.4f}")
print(f"AUC-ROC   : {xgb_auc:.4f}")
# ============================================================
# 16. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        accuracy,
        rf_accuracy,
        xgb_accuracy
    ],
    "Precision": [
        precision,
        rf_precision,
        xgb_precision
    ],
    "Recall": [
        recall,
        rf_recall,
        xgb_recall
    ],
    "F1-Score": [
        f1,
        rf_f1,
        xgb_f1
    ],
    "AUC-ROC": [
        auc,
        rf_auc,
        xgb_auc
    ]
})

print("\n========================================")
print("             MODEL COMPARISON")
print("========================================")

print(results.to_string(index=False))
# ============================================
# SAVE BEST MODEL AND PREPROCESSOR
# ============================================

import joblib
import os

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save Logistic Regression model
joblib.dump(logistic_model, "model/logistic_regression_model.pkl")

# Save preprocessor
joblib.dump(preprocessor, "model/preprocessor.pkl")

print("\n============================================")
print("BEST MODEL SAVED SUCCESSFULLY")
print("============================================")
print("Model      : Logistic Regression")
print("Model file : model/logistic_regression_model.pkl")
print("Preprocess : model/preprocessor.pkl")