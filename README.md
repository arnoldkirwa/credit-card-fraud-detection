# Credit Card Fraud Detection

## Project Overview

Credit card fraud causes financial losses and reduces trust in digital payment systems. This project uses machine learning to identify potentially fraudulent transactions in a highly imbalanced dataset.

The project follows the **CRISP-DM methodology**, covering business understanding, data understanding, data preparation, modeling, evaluation, and deployment.

## Business Problem

Fraudulent transactions represent a small proportion of all transactions, making fraud detection a challenging imbalanced classification problem. The goal is to identify as many fraudulent transactions as possible while controlling false alarms.

## Project Objectives

- Analyze transaction data and understand class imbalance.
- Prepare data for machine learning.
- Compare classification models.
- Evaluate fraud detection using precision, recall, F1-score, and Average Precision.
- Deploy a model through an interactive Streamlit web application.
- Support individual transaction predictions and batch CSV analysis.

## Dataset

The project uses the Credit Card Fraud Detection dataset, containing anonymized European cardholder transactions.

- **Original records:** 284,807
- **Records after duplicate removal:** 283,726
- **Input features:** 30
- **Target:** `Class` — 0 for legitimate transactions and 1 for fraudulent transactions
- **Fraud cases after duplicate removal:** 473

The dataset contains anonymized features `V1`–`V28`, along with `Time` and `Amount`.

The dataset is not included in this repository because of its size. Obtain it from the original dataset source used for the project and place `creditcard.csv` inside the `data/` folder.

## Methodology

The project follows CRISP-DM:

1. **Business Understanding:** Define the fraud detection problem and project objectives.
2. **Data Understanding:** Inspect the dataset, class distribution, and data quality.
3. **Data Preparation:** Remove duplicate records and prepare features and stratified training and testing sets.
4. **Modeling:** Compare Logistic Regression, Decision Tree, Random Forest, and XGBoost.
5. **Evaluation:** Assess model performance using fraud-focused metrics and a selected decision threshold.
6. **Deployment:** Save the final model and build an interactive Streamlit application.

## Model Evaluation

The deployed Random Forest model was evaluated on the held-out test set at a decision threshold of 0.30.

| Metric | Result |
|---|---:|
| Precision | 64.17% |
| Recall | 81.05% |
| F1-score | 71.63% |
| Average Precision | 80.69% |
| True positives | 77 |
| False positives | 43 |
| False negatives | 18 |
| True negatives | 56,608 |

**Interpretation:** The model detected 77 of the 95 fraudulent transactions in the test set and missed 18. It also flagged 43 legitimate transactions for potential review.

The threshold reflects a trade-off between detecting fraud and generating false alarms. These results are from a held-out test set and do not guarantee the same performance in production.

## Streamlit Application
**Live Demo:** [Credit Card Fraud Detection App](https://credit-card-fraud-detection-pezfcmmas5xay83dycsy2e.streamlit.app/)

Try the deployed application to explore individual transaction predictions and batch CSV fraud detection.

The application supports:

- Individual transaction analysis.
- Estimated fraud probability.
- Fraud classification using a 0.30 decision threshold.
- Batch transaction analysis using CSV uploads.
- Downloadable CSV prediction results.

Predictions are intended to support further investigation and should not replace human review.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Jupyter Notebook
- Streamlit
- Joblib
- Git and GitHub

## Project Structure

```text
credit-card-fraud-detection/
├── app/
│   └── app.py
├── data/
│   └── creditcard.csv
├── models/
│   └── credit_card_fraud_random_forest.joblib
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset and local test CSV files are excluded from Git tracking.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/arnoldkirwa/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the original dataset at:

```text
data/creditcard.csv
```

### 5. Run the application

```bash
streamlit run app/app.py
```

The application should open in your browser. The saved model must be present in `models/` for predictions to work.

## Limitations and Future Improvements

- The dataset contains anonymized features, limiting direct business interpretation.
- Fraud patterns may change over time, requiring monitoring and retraining.
- False positives may lead to unnecessary transaction reviews.
- The 0.30 threshold is a project choice and should be validated against actual business costs before operational use.
- Future improvements could include model monitoring, threshold calibration, explainability, and testing on newer transaction data.

## Author

**Arnold Kirwa**

GitHub: [arnoldkirwa](https://github.com/arnoldkirwa)
