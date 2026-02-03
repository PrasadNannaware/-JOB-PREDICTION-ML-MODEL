# 
import streamlit as st
import joblib
import numpy as np

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Job Package Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- Load Model ----------------
model = joblib.load("regression_model.joblib")

# ---------------- Header Image ----------------
st.image(
    "https://images.unsplash.com/photo-1523050854058-8df90110c9f1",
    use_container_width=True
)

# ---------------- Title & Description ----------------
st.markdown(
    "<h1 style='text-align:center; color:#2E86C1;'>🎓 Job Package Prediction Based on CGPA</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; font-size:16px;'>"
    "This Machine Learning application predicts the <b>expected job package</b> "
    "based on a student's <b>CGPA</b> using a <b>Linear Regression model</b>."
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- Input Section ----------------
st.subheader("📊 Enter Your Academic Details")

cgpa = st.number_input(
    "CGPA (Out of 10)",
    min_value=0.0,
    max_value=10.0,
    step=0.1,
    help="Enter your CGPA to predict your expected job package"
)

# ---------------- Prediction ----------------
if st.button("🚀 Predict Job Package"):
    input_data = np.array([[cgpa]])
    prediction = model.predict(input_data)

    predicted_value = max(prediction.item(), 0)

    st.success(f"💼 **Predicted Job Package:** ₹ {predicted_value:,.2f} LPA")
    st.balloons()

# ---------------- Project Highlights ----------------
st.divider()
st.markdown(
    """
    ### 📌 Project Highlights
    - 📚 Algorithm Used: **Linear Regression**
    - 🧠 Machine Learning Model
    - 🐍 Programming Language: **Python**
    - 🌐 Frontend: **Streamlit**
    - 📊 Input Feature: **CGPA**
    """
)

# ---------------- Footer ----------------
st.markdown(
    "<hr>"
    "<p style='text-align:center; font-size:13px;'>"
    "Developed by <b>Saurabh Shinde</b><br>"
    "Machine Learning Project"
    "</p>",
    unsafe_allow_html=True
)
