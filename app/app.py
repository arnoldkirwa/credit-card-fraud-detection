
# 6.12 Credit Card Fraud Detection Web Application

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_card_fraud_random_forest.joblib"
)

THRESHOLD = 0.3


@st.cache_resource
def load_model():
    """Load the saved fraud detection pipeline."""
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Application interface
# --------------------------------------------------

st.title("🛡️ Credit Card Fraud Detection")
st.write(
    "Enter the transaction features below to estimate "
    "whether the transaction may be fraudulent."
)

st.info(
    "This is a machine learning prototype. "
    "Predictions should support investigation, not replace "
    "human review."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()


try:
    model = load_model()
    feature_names = list(model.feature_names_in_)
except Exception as error:
    st.error(
        "The model could not be loaded. Check that the saved model and library versions are compatible."
    )
    st.exception(error)
    st.stop()




# --------------------------------------------------
# Batch Transaction Upload
# --------------------------------------------------

st.divider()
st.subheader("Batch Fraud Detection")
st.write(
    "Upload a CSV file containing multiple transactions. "
    "The file must contain the same 30 features used to "
    "train the model."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    key="batch_transaction_upload"
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        st.write("Uploaded data preview:")
        st.dataframe(batch_df.head())

        missing_features = [
            feature for feature in feature_names
            if feature not in batch_df.columns
        ]

        extra_features = [
            column for column in batch_df.columns
            if column not in feature_names
        ]

        if missing_features:
            st.error(
                f"Missing required features: {missing_features}"
            )

        elif extra_features:
            st.error(
                f"Unexpected columns found: {extra_features}. "
                "Upload only the 30 model features."
            )

        elif batch_df.empty:
            st.error("The uploaded CSV contains no transactions.")

        else:
            # Enforce the model's expected feature order
            batch_features = batch_df[feature_names].copy()

            # Validate numeric data
            batch_features = batch_features.apply(
                pd.to_numeric, errors="coerce"
            )

            if batch_features.isna().any().any():
                st.error(
                    "The CSV contains missing or non-numeric "
                    "feature values. Please correct the file."
                )

            elif not batch_features.map(
                lambda value: __import__("math").isfinite(value)
            ).all().all():
                st.error(
                    "The CSV contains infinite or invalid "
                    "numeric values."
                )

            else:
                # Predict fraud probabilities
                probabilities = model.predict_proba(
                    batch_features
                )[:, 1]

                predictions = (
                    probabilities >= THRESHOLD
                ).astype(int)

                # Prepare downloadable results
                results_df = batch_df.copy()
                results_df["fraud_probability"] = probabilities
                results_df["predicted_class"] = predictions
                results_df["prediction_result"] = [
                    (
                        "Potentially fraudulent"
                        if prediction == 1
                        else "Not flagged as fraudulent"
                    )
                    for prediction in predictions
                ]

                st.success(
                    f"Successfully analyzed "
                    f"{len(results_df)} transactions."
                )

                st.dataframe(results_df)

                st.download_button(
                    label="Download Prediction Results",
                    data=results_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

    except Exception as error:
        st.error(
            "Unable to process the uploaded CSV. "
            "Check that it is a valid CSV file."
        )
        st.exception(error)

# --------------------------------------------------
# Transaction input
# --------------------------------------------------

st.subheader("Transaction Details")

st.caption(
    "Provide all 30 numerical features in the same order "
    "used during model training. For this dataset, the "
    "features are Time, V1–V28, and Amount."
)

feature_names = list(model.feature_names_in_)

with st.form("transaction_form"):
    values = {}

    for feature in feature_names:
        default_value = 0.0

        values[feature] = st.number_input(
            label=feature,
            value=default_value,
            format="%.6f",
            key=f"feature_{feature}"
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary"
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:
    transaction = pd.DataFrame(
        [values],
        columns=feature_names
    )

    try:
        fraud_probability = float(
            model.predict_proba(transaction)[0, 1]
        )

        predicted_class = int(
            fraud_probability >= THRESHOLD
        )

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Estimated Fraud Probability",
                f"{fraud_probability:.2%}"
            )

        with col2:
            st.metric(
                "Decision Threshold",
                f"{THRESHOLD:.0%}"
            )

        if predicted_class == 1:
            st.error(
                "Potentially fraudulent transaction — "
                "further review is recommended."
            )
        else:
            st.success(
                "Not flagged as fraudulent at the selected "
                "threshold. This does not guarantee the "
                "transaction is legitimate."
            )

    except Exception as error:
        st.error("Prediction failed. Please check the input.")
        st.exception(error)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Model: Random Forest | "
    "Dataset: Credit Card Fraud Detection | "
    "Decision threshold: 0.30"
)
