import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="centered"
)

# Load models
scaler = pickle.load(open("scaler.pkl", "rb"))
svm_linear = pickle.load(open("svm_linear.pkl", "rb"))
svm_poly   = pickle.load(open("svm_poly.pkl", "rb"))
svm_rbf    = pickle.load(open("svm_rbf.pkl", "rb"))

# Title
st.title("💳 Smart Loan Approval System")
st.write(
    "This system uses **Support Vector Machines (SVM)** "
    "to predict whether a loan should be approved or rejected."
)

# Sidebar inputs
st.sidebar.header("Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0)

credit_history = st.sidebar.selectbox(
    "Credit History", ["Yes", "No"]
)

employment = st.sidebar.selectbox(
    "Employment Status", ["Salaried", "Self-Employed"]
)

property_area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

# Encode inputs
credit_history = 1 if credit_history == "Yes" else 0
employment = 1 if employment == "Salaried" else 0
property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

# Kernel selection
st.subheader("Select SVM Kernel")

kernel = st.radio(
    "Choose model:",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

model = (
    svm_linear if kernel == "Linear SVM"
    else svm_poly if kernel == "Polynomial SVM"
    else svm_rbf
)

# Predict
if st.button("Check Loan Eligibility"):

    input_data = np.array([[
        income,
        loan_amount,
        credit_history,
        employment,
        property_area
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max()

    if prediction == 1:
        st.success("✅ Loan Approved")
        decision = "likely to repay the loan"
    else:
        st.error("❌ Loan Rejected")
        decision = "unlikely to repay the loan"

    st.write(f"**Kernel Used:** {kernel}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.info(
        f"Based on credit history and income pattern, "
        f"the applicant is **{decision}**."
    )
