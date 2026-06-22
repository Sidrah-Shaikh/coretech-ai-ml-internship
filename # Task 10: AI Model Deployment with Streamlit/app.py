%%writefile app.py

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Student Pass Prediction")

st.write(
    "This app uses a Machine Learning model to predict whether a student will pass based on Age, Study Hours, and Attendance."
)

age = st.number_input("Age",18,30)
study = st.number_input("Study Hours",1,10)
attendance = st.number_input("Attendance",50,100)

if st.button("Predict"):
    data = pd.DataFrame(
        [[age,study,attendance]],
        columns=["Age","StudyHours","Attendance"]
    )

    result = model.predict(data)

    st.success(f"Prediction: {result[0]}")
