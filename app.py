import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Smart Loan Approval System")

# ---------------------------
# Load scaler & models
# ---------------------------
scaler = pickle.load(open("scaler.pkl", "rb"))
svm_linear = pickle.load(open("svm_linear.pkl", "rb"))
svm_poly   = pickle.load(open("svm_poly.pkl", "rb"))
svm_rbf    = pickle.load(open("svm_rbf.pkl", "rb"))

# ---------------------------
# DEBUG INFO (THIS IS KEY)
# ---------------------------
st.sidebar.markdown("### 🔍 Debug Info")
st.sidebar.write("Scaler expects features:", scaler.n_features_in_)

# ---------------------------
# UI
# ---------------------------
st.title("💳 Smart Loan Approval System")
st.write("Loan approval prediction using Support Vector Machines")

st.sidebar.header("Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0)

credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Encode
credit_history = 1 if credit_history == "Yes" else 0
employment = 1 if employment == "Salaried" else 0
property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

# Feature vector (EXACTLY 5 FEATURES)
input_data = np.array([[
    income,
    loan_amount,
    credit_history,
    employment,
    property_area
]])

st.sidebar.write("Input feature count:", input_data.shape[1])

# Model selection
kernel = st.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

model = (
    svm_linear if kernel == "Linear SVM"
    else svm_poly if kernel == "Polynomial SVM"
    else svm_rbf
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Check Loan Eligibility"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max()

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Kernel Used:** {kernel}")
    st.write(f"**Confidence:** {confidence:.2f}")
