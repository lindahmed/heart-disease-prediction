import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

# =====================================
# STEP 1: LOAD THE TRAINED MODEL
# =====================================
model = joblib.load("notebooks/best_model_tuned.pkl")

# =====================================
# STEP 2: SET UP STREAMLIT APP CONFIG
# =====================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction App")
st.write("Fill in the details below and get an instant prediction!")

# =====================================
# STEP 3: USER INPUT FIELDS
# =====================================
def user_input_features():
    # Define input fields for the required features
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
    thalach = st.slider("Max Heart Rate Achieved (thalach)", 70, 220, 150)
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (chol)", 100, 600, 200)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    age = st.slider("Age", 18, 100, 40)
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)

    # Store data in correct feature order
    data = {
        "cp": cp,
        "sex": 1 if sex == "Male" else 0,
        "exang": exang,
        "ca": ca,
        "thal": thal,
        "thalach": thalach,
        "trestbps": trestbps,
        "chol": chol,
        "slope": slope,
        "age": age,
        "oldpeak": oldpeak
    }

    return pd.DataFrame(data, index=[0])


    return pd.DataFrame(data, index=[0])

# Get the user inputs
input_df = user_input_features()

st.subheader("🔹 Your Input Data")
st.write(input_df)

# =====================================
# STEP 4: MAKE PREDICTION
# =====================================
prediction = model.predict(input_df)
prediction_prob = model.predict_proba(input_df)

# Display prediction
st.subheader("🔹 Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ High Risk of Heart Disease!")
else:
    st.success("✅ Low Risk of Heart Disease")

# Display probabilities
st.subheader("🔹 Prediction Probability")
st.write(f"**Low Risk:** {prediction_prob[0][0]*100:.2f}%")
st.write(f"**High Risk:** {prediction_prob[0][1]*100:.2f}%")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.markdown("Developed with ❤️ using **Streamlit** & **Machine Learning**")
