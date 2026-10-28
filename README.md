# Explainable AI-Based Credit Risk Prediction System

## Project Overview

The **Explainable AI-Based Credit Risk Prediction System** is a machine-learning web application designed to predict credit risk from applicant information. It combines a trained Logistic Regression model with explainability techniques to make predictions easier to understand.

The system provides a web interface for submitting applicant details, viewing risk predictions, exploring model explanations, and reviewing saved prediction history.

## Key Features

- **Credit Risk Prediction:** Predicts credit risk from applicant attributes.
- **Machine Learning:** Uses a trained Logistic Regression model and a preprocessing pipeline.
- **Explainable AI:** Integrates SHAP and LIME to help interpret model predictions.
- **Prediction History:** Stores prediction records using an SQLite database.
- **Analytics Dashboard:** Displays total predictions, high-risk cases, and low-risk cases.
- **Web Interface:** Provides a browser-based interface built with HTML, CSS, and JavaScript.
- **Flask Backend:** Connects the frontend, machine-learning model, and database.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Backend | Flask |
| Machine learning | Scikit-learn |
| Data processing | Pandas, NumPy |
| Model interpretation | SHAP, LIME |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Model persistence | Joblib |

## Project Structure

```text
credit-risk-prediction/
├── app.py
├── database.py
├── model/
│   ├── logistic_regression_model.pkl
│   └── preprocessor.pkl
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── README.md
├── requirements.txt
└── .gitignore
```

The SQLite database is created locally when the application initializes. The exact files in the repository may vary depending on the final project configuration.

## Installation and Setup

### Prerequisites

- Python 3.12 or another Python version compatible with the project's dependencies
- Git
- Visual Studio Code or another code editor

### 1. Clone the repository

```bash
git clone https://github.com/Praveen-S90/-credit-risk-prediction.git
cd -credit-risk-prediction
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local address shown in the Flask terminal, typically:

```text
http://127.0.0.1:5000
```

**Note:** The trained model files must be available in the expected `model/` directory. The database is initialized by the application.

## How to Use

1. Start the Flask application.
2. Open the website in a browser.
3. Enter the applicant information requested by the form.
4. Submit the form to generate a prediction.
5. Review the predicted risk and probability results.
6. Examine the SHAP and LIME explanations, when available.
7. View saved predictions in the history section.
8. Review the dashboard statistics.

## Explainable AI

SHAP and LIME are used to help interpret the model's output by showing how input features contribute to a prediction.

These explanations describe model behavior; they do not establish that a feature causes credit default or that the prediction is necessarily correct.

## Database

The application uses SQLite to store prediction-history records, including selected applicant attributes, risk results, probabilities, and timestamps.

The database is intended for local application use. Do not publish real applicant data or sensitive financial information in the repository.

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- Model outputs are estimates and may contain errors.
- Explainability methods provide interpretations, not guarantees of correctness.
- The system is an academic project and should not be used as the sole basis for real lending decisions.
- Fairness, privacy, security, and model performance require further evaluation before any real-world deployment.

## Future Enhancements

- Improve model evaluation and compare multiple algorithms.
- Add additional model-performance and fairness metrics.
- Improve authentication and access control.
- Add exportable reports and richer dashboard visualizations.
- Evaluate the model on more representative data.

## Academic Purpose

This project was developed for academic learning and demonstration of machine learning, web development, database integration, and Explainable AI techniques.

## License

Add an appropriate license before redistributing this project. If no license is provided, reuse permissions are not automatically granted.
