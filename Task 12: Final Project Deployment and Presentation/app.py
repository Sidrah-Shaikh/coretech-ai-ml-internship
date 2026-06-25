
import streamlit as st
import pandas as pd
import pickle, json, os
from PIL import Image

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="wide")

@st.cache_resource
def load_artifacts():
    model    = pickle.load(open("models/model.pkl",    "rb"))
    scaler   = pickle.load(open("models/scaler.pkl",   "rb"))
    encoders = pickle.load(open("models/encoders.pkl", "rb"))
    return model, scaler, encoders

@st.cache_data
def load_results():
    return json.load(open("models/eval_results.json"))

model, scaler, encoders = load_artifacts()
results = load_results()

st.sidebar.title("🚢 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home & Predict", "📊 EDA Charts", "📈 Model Evaluation"])

if page == "🏠 Home & Predict":
    st.title("🚢 Titanic Survival Predictor")
    st.markdown("Enter passenger details below to predict survival probability.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Passenger Information")
        pclass = st.selectbox("Passenger Class", options=[1,2,3],
            format_func=lambda x: f"Class {x} - {['First','Second','Third'][x-1]}")
        sex = st.radio("Sex", options=["male","female"], horizontal=True)
        age = st.slider("Age (years)", min_value=1, max_value=80, value=28)
        fare = st.number_input("Ticket Fare", min_value=0.0, max_value=600.0, value=32.0, step=0.5)
    with col2:
        st.subheader("👨‍👩‍👧 Family & Boarding")
        sibsp = st.number_input("Siblings / Spouses aboard", min_value=0, max_value=8, value=0)
        parch = st.number_input("Parents / Children aboard", min_value=0, max_value=6, value=0)
        embarked = st.selectbox("Port of Embarkation", options=["S","C","Q"],
            format_func=lambda x: {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[x])
    st.markdown("---")
    if st.button("🔮 Predict Survival", use_container_width=True, type="primary"):
        input_df = pd.DataFrame([{"Pclass":pclass,"Sex":sex,"Age":float(age),
                                   "SibSp":sibsp,"Parch":parch,"Fare":float(fare),"Embarked":embarked}])
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
        input_df[["Age","Fare","SibSp","Parch"]] = scaler.transform(input_df[["Age","Fare","SibSp","Parch"]])
        prob = model.predict_proba(input_df)[0][1]
        pred = int(prob >= 0.5)
        st.markdown("### 🎯 Prediction Result")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.success("✅ SURVIVED") if pred==1 else st.error("❌ DID NOT SURVIVE")
        with r2:
            st.metric("Survival Probability", f"{prob*100:.1f}%")
        with r3:
            st.metric("Model Accuracy", f"{results['accuracy']*100:.1f}%")
        st.progress(float(prob))

elif page == "📊 EDA Charts":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    titles = ["Survival Count","Survival by Gender","Age Distribution",
              "Survival by Class","Fare by Survival","Correlation Heatmap"]
    charts = sorted([f for f in os.listdir("eda_charts") if f.endswith(".png")])
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j in range(2):
            if i+j < len(charts):
                cols[j].image(Image.open(f"eda_charts/{charts[i+j]}"), caption=titles[i+j], use_column_width=True)

elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("✅ Accuracy", f"{results['accuracy']*100:.2f}%")
    c2.metric("📊 ROC-AUC",  f"{results['roc_auc']:.4f}")
    st.markdown("### Classification Report")
    st.code(results["report"], language="text")
    st.info("**Algorithm:** Random Forest | **Trees:** 200 | **Max Depth:** 8 | **Split:** 80/20")
