import streamlit as st
import pandas as pd
import joblib


# =========================
# Load Model & Preprocessing
# =========================

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")



st.title("🍄 Mushroom Classification")
st.write("Predict whether a mushroom is edible or poisonous.")


st.subheader("Enter Mushroom Features")

feature_columns = [col for col in encoders if col != "class"]

features = {}

for col in feature_columns:
    features[col] = st.selectbox(
        col,
        encoders[col].classes_
    )



if st.button("Predict"):

    input_data = pd.DataFrame(
        [features],
        columns=feature_columns
    )

    # Encode categorical values
    for col in input_data.columns:
        input_data[col] = encoders[col].transform(input_data[col])

    # Scaling
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("🍄 The mushroom is Edible")
    else:
        st.error("☠️ The mushroom is Poisonous")