from flask import Flask, render_template, request, jsonify

import pandas as pd
import numpy as np
import joblib
import shap

from lime.lime_tabular import LimeTabularExplainer
from datetime import datetime
from database import initialize_database, save_prediction, get_prediction_history


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
initialize_database()


# =========================================================
# LOAD MODEL AND PREPROCESSOR
# =========================================================

model = joblib.load("model/logistic_regression_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")


# =========================================================
# DATASET COLUMN NAMES
# =========================================================

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


# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv(
    "data/german.data",
    sep=r"\s+",
    header=None,
    names=columns
)


# =========================================================
# SEPARATE FEATURES
# =========================================================

X = data.drop("credit_risk", axis=1)


# =========================================================
# PROCESS DATA FOR EXPLAINABILITY
# =========================================================

X_processed = preprocessor.transform(X)

if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

X_processed = np.asarray(
    X_processed,
    dtype=float
)


# =========================================================
# FEATURE NAMES
# =========================================================

feature_names = preprocessor.get_feature_names_out()


def clean_feature_name(name):

    name = str(name)

    if name.startswith("num__"):
        return name.replace("num__", "")

    if name.startswith("cat__"):

        name = name.replace("cat__", "")

        parts = name.rsplit("_", 1)

        if len(parts) == 2:
            return f"{parts[0]} ({parts[1]})"

        return name

    return name


display_feature_names = [
    clean_feature_name(name)
    for name in feature_names
]


# =========================================================
# SHAP EXPLAINER
# =========================================================

shap_explainer = shap.LinearExplainer(
    model,
    X_processed
)


# =========================================================
# LIME EXPLAINER
# =========================================================

lime_explainer = LimeTabularExplainer(
    X_processed,
    feature_names=display_feature_names,
    class_names=[
        "Low Risk",
        "High Risk"
    ],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )
@app.route("/history", methods=["GET"])
def prediction_history():
    try:
        history = get_prediction_history(limit=100)
        return jsonify({"history": history})
    except Exception as error:
        print("DATABASE HISTORY ERROR:", error)
        return jsonify({"error": "Could not load prediction history"}), 500


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =================================================
        # GET DATA
        # =================================================

        input_data = request.get_json()

        if not input_data:

            return jsonify({
                "error": "No input data received"
            }), 400


        # =================================================
        # CREATE APPLICANT DATAFRAME
        # =================================================

        applicant = pd.DataFrame([
            input_data
        ])


        # =================================================
        # CORRECT COLUMN ORDER
        # =================================================

        applicant = applicant[
            X.columns
        ]


        # =================================================
        # NUMERICAL COLUMNS
        # =================================================

        numerical_columns = [

            "duration",

            "credit_amount",

            "installment_rate",

            "residence_since",

            "age",

            "existing_credits",

            "num_dependents"

        ]


        for col in numerical_columns:

            applicant[col] = pd.to_numeric(
                applicant[col],
                errors="coerce"
            )


        # =================================================
        # VALIDATE NUMERICAL INPUT
        # =================================================

        if applicant[
            numerical_columns
        ].isnull().any().any():

            return jsonify({

                "error":
                "Please enter valid numerical values."

            }), 400


        # =================================================
        # PREPROCESS
        # =================================================

        applicant_processed = (
            preprocessor.transform(applicant)
        )


        if hasattr(
            applicant_processed,
            "toarray"
        ):

            applicant_processed = (
                applicant_processed.toarray()
            )


        applicant_processed = np.asarray(
            applicant_processed,
            dtype=float
        )


        # =================================================
        # CHECK INVALID VALUES
        # =================================================

        if not np.isfinite(
            applicant_processed
        ).all():

            return jsonify({

                "error":
                "Invalid values generated during preprocessing."

            }), 500


        # =================================================
        # MODEL PREDICTION
        # =================================================

        prediction = model.predict(
            applicant_processed
        )[0]


        probabilities = model.predict_proba(
            applicant_processed
        )[0]


        # =================================================
        # PROBABILITIES
        # =================================================

        low_risk_probability = float(
            probabilities[0] * 100
        )

        high_risk_probability = float(
            probabilities[1] * 100
        )


        low_risk_probability = round(
            low_risk_probability,
            2
        )

        high_risk_probability = round(
            high_risk_probability,
            2
        )


        # =================================================
        # RISK SCORE
        # =================================================

        risk_score = high_risk_probability


        # =================================================
        # RISK LEVEL
        # =================================================

        if risk_score < 30:

            risk_level = "LOW"

            risk_message = (
                "The applicant shows a relatively "
                "low probability of credit default."
            )

        elif risk_score < 60:

            risk_level = "MEDIUM"

            risk_message = (
                "The applicant has a moderate "
                "level of credit risk."
            )

        else:

            risk_level = "HIGH"

            risk_message = (
                "The applicant shows a relatively "
                "high probability of credit default."
            )


        # =================================================
        # SHAP EXPLANATION
        # =================================================

        shap_explanation = []


        try:

            shap_values = shap_explainer(
                applicant_processed
            )


            values = np.asarray(
                shap_values.values
            )


            if values.ndim == 3:

                values = values[
                    0,
                    :,
                    1
                ]

            elif values.ndim == 2:

                values = values[0]

            elif values.ndim == 1:

                values = values

            else:

                values = np.zeros(
                    len(display_feature_names)
                )


            if len(values) != len(
                display_feature_names
            ):

                values = np.resize(
                    values,
                    len(display_feature_names)
                )


            shap_pairs = []


            for feature, value in zip(
                display_feature_names,
                values
            ):

                try:

                    value = float(value)


                    if not np.isfinite(
                        value
                    ):

                        continue


                    shap_pairs.append({

                        "feature": feature,

                        "shap_value": value

                    })


                except Exception:

                    continue


            shap_pairs.sort(

                key=lambda x:
                abs(x["shap_value"]),

                reverse=True
            )


            shap_explanation = (
                shap_pairs[:8]
            )


        except Exception as shap_error:

            print(
                "SHAP ERROR:",
                shap_error
            )

            shap_explanation = []


        # =================================================
        # LIME EXPLANATION
        # =================================================

        lime_explanation = []


        try:

            lime_result = (
                lime_explainer.explain_instance(

                    applicant_processed[0],

                    model.predict_proba,

                    num_features=8

                )
            )


            for feature, weight in (
                lime_result.as_list()
            ):

                try:

                    weight = float(
                        weight
                    )


                    if not np.isfinite(
                        weight
                    ):

                        continue


                    lime_explanation.append({

                        "feature":
                        str(feature),

                        "weight":
                        weight

                    })


                except Exception:

                    continue


        except Exception as lime_error:

            print(
                "LIME ERROR:",
                lime_error
            )

            lime_explanation = []


        # =================================================
        # MODEL COMPARISON
        # =================================================

        model_comparison = [

            {
                "model":
                "Logistic Regression",

                "accuracy":
                74.5,

                "precision":
                55.3,

                "recall":
                78.3,

                "f1":
                64.8,

                "auc":
                79.7,

                "selected":
                True
            },

            {
                "model":
                "Random Forest",

                "accuracy":
                76.0,

                "precision":
                64.3,

                "recall":
                45.0,

                "f1":
                52.9,

                "auc":
                78.8,

                "selected":
                False
            },

            {
                "model":
                "XGBoost",

                "accuracy":
                76.0,

                "precision":
                61.1,

                "recall":
                55.0,

                "f1":
                57.9,

                "auc":
                77.8,

                "selected":
                False
            }

        ]


        # =================================================
        # APPLICANT SUMMARY
        # =================================================

        applicant_summary = {

            "age":
            int(applicant["age"].iloc[0]),

            "credit_amount":
            float(
                applicant[
                    "credit_amount"
                ].iloc[0]
            ),

            "duration":
            int(
                applicant[
                    "duration"
                ].iloc[0]
            ),

            "employment":
            str(
                applicant[
                    "employment"
                ].iloc[0]
            ),

            "housing":
            str(
                applicant[
                    "housing"
                ].iloc[0]
            ),

            "credit_history":
            str(
                applicant[
                    "credit_history"
                ].iloc[0]
            ),

            "existing_credits":
            int(
                applicant[
                    "existing_credits"
                ].iloc[0]
            )

        }


        # =================================================
        # TIMESTAMP
        # =================================================

        prediction_time = (
            datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )
        )


        # =================================================
        # FINAL RESPONSE
        # =================================================

        response = {

            # Prediction
            "prediction":
            int(prediction),

            # Probabilities
            "low_risk_probability":
            low_risk_probability,

            "high_risk_probability":
            high_risk_probability,

            # Risk
            "risk_score":
            risk_score,

            "risk_level":
            risk_level,

            "risk_message":
            risk_message,

            # Explainability
            "shap_explanation":
            shap_explanation,

            "lime_explanation":
            lime_explanation,

            # Applicant
            "applicant_summary":
            applicant_summary,

            # Models
            "model_comparison":
            model_comparison,

            # Time
            "prediction_time":
            prediction_time

        }


        # =================================================
        # TERMINAL OUTPUT
        # =================================================

        print(
            "\n=============================="
        )

        print(
            "PREDICTION:",
            prediction
        )

        print(
            "RISK LEVEL:",
            risk_level
        )

        print(
            "RISK SCORE:",
            risk_score,
            "%"
        )

        print(
            "LOW RISK:",
            low_risk_probability,
            "%"
        )

        print(
            "HIGH RISK:",
            high_risk_probability,
            "%"
        )

        print(
            "\nSHAP:"
        )

        print(
            shap_explanation
        )

        print(
            "\nLIME:"
        )

        print(
            lime_explanation
        )

        print(
            "==============================\n"
        )


        
        save_prediction(input_data, response)

        return jsonify(
            response
        )



    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as error:

        print(
            "PREDICTION ERROR:",
            error
        )

        return jsonify({

            "error":
            str(error)

        }), 500


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )