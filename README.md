# 💳 Explainable AI-Based Credit Risk Prediction System

### An Interpretable Machine Learning Approach for Transparent Credit Scoring

## 📌 About the Project

The Explainable AI-Based Credit Risk Prediction System is a machine learning web application that predicts the credit risk of loan applicants using applicant information. It aims to make credit risk assessment more transparent by combining machine learning predictions with explainable AI techniques.

Unlike traditional prediction systems that provide only a result, this project uses **SHAP and LIME** to help explain the factors influencing individual predictions.

## 🎯 Objectives

- Predict the credit risk of loan applicants.
- Estimate the probability of high-risk and low-risk outcomes.
- Improve transparency using explainable AI techniques.
- Present predictions through an interactive web interface.
- Compare machine learning models using evaluation metrics.

## ✨ Features

- 🤖 Machine learning-based credit risk prediction
- 📊 Credit risk probability visualization
- 🔍 SHAP-based feature explanations
- 🧠 LIME-based individual prediction explanations
- 📈 Model performance comparison
- 🌐 Interactive web interface built with Flask
- 📋 Applicant risk assessment summary

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** Scikit-learn, Logistic Regression
- **Explainable AI:** SHAP, LIME
- **Data Processing:** Pandas, NumPy
- **Dataset:** German Credit Data

## 🏗️ Project Structure

```text
credit-risk-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── german.data
├── model/
│   ├── logistic_regression_model.pkl
│   └── preprocessor.pkl
├── explainability/
│   ├── shap_explainer.py
│   └── lime_explainer.py
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

*Note: The structure above is illustrative. Update it if your actual filenames differ.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Praveen-S90/-credit-risk-prediction.git
cd -credit-risk-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open the application

Visit this address in your browser:

```text
http://127.0.0.1:5000
```

## 📊 Machine Learning and Explainability

The project uses Logistic Regression for credit risk prediction. Model performance can be evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

- **SHAP:** Explains how input features contribute to a prediction.
- **LIME:** Provides a local explanation of an individual prediction.

## 🎓 Project Purpose

This project was developed as an academic major project to explore machine learning, interpretable AI, and transparent credit risk assessment.

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. Its predictions should not be used as the sole basis for real-world lending decisions.

## 👨‍💻 Author

**Praveen S90**

GitHub: [@Praveen-S90](https://github.com/Praveen-S90)

---

⭐ If you find this project interesting, consider starring the repository!
