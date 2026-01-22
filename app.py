import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load models and scaler
# -------------------------------
linear_model = pickle.load(open("svm_linear.pkl", "rb"))
poly_model   = pickle.load(open("svm_poly.pkl", "rb"))
rbf_model    = pickle.load(open("svm_rbf.pkl", "rb"))
scaler       = pickle.load(open("scaler.pkl", "rb"))


# -------------------------------
# App Title & Description
# -------------------------------
st.title("💳 Smart Loan Approval System")

st.write(
    "This system uses **Support Vector Machines (SVM)** to predict whether a "
    "loan should be **Approved or Rejected** based on applicant details."
)

# -------------------------------
# Sidebar – User Inputs
# -------------------------------
st.sidebar.header("Applicant Details")

applicant_income = st.sidebar.number_input(
    "Applicant Income", min_value=0.0, step=100.0
)

loan_amount = st.sidebar.number_input(
    "Loan Amount", min_value=0.0, step=100.0
)

credit_history = st.sidebar.selectbox(
    "Credit History",
    options=["Yes", "No"]
)

employment_status = st.sidebar.selectbox(
    "Employment Status",
    options=["Salaried", "Self-Employed"]
)

property_area = st.sidebar.selectbox(
    "Property Area",
    options=["Urban", "Semiurban", "Rural"]
)

# Encode inputs
credit_history = 1 if credit_history == "Yes" else 0
employment_status = 1 if employment_status == "Salaried" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# -------------------------------
# Model Selection
# -------------------------------
st.subheader("Choose SVM Kernel")

kernel_choice = st.radio(
    "Select the model kernel:",
    ("Linear SVM", "Polynomial SVM", "RBF SVM")
)

# Choose model
if kernel_choice == "Linear SVM":
    model = linear_model
elif kernel_choice == "Polynomial SVM":
    model = poly_model
else:
    model = rbf_model

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Check Loan Eligibility"):

    input_data = np.array([[
        applicant_income,
        loan_amount,
        credit_history,
        employment_status,
        property_area
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    # Confidence (optional)
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(input_scaled).max()
    else:
        confidence = None

    # -------------------------------
    # Output Section
    # -------------------------------
    if prediction == 1:
        st.success("✅ Loan Approved")
        decision_text = "likely to repay the loan"
    else:
        st.error("❌ Loan Rejected")
        decision_text = "unlikely to repay the loan"

    # Optional details
    st.write(f"**Kernel Used:** {kernel_choice}")

    if confidence is not None:
        st.write(f"**Model Confidence:** {confidence:.2f}")

    # -------------------------------
    # Business Explanation
    # -------------------------------
    st.info(
        f"Based on **credit history** and **income pattern**, "
        f"the applicant is **{decision_text}**."
    )
