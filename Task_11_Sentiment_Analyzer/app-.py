
import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# --- UI ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")
st.title("💬 Client Feedback Sentiment Analyzer")
st.markdown("Enter customer feedback below to predict sentiment.")

feedback = st.text_area("📝 Enter Feedback Here", height=150)

if st.button("Analyze Sentiment"):
    if feedback.strip() == "":
        st.warning("Please enter some feedback text.")
    else:
        cleaned = clean_text(feedback)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        confidence = max(proba) * 100

        if prediction == "Positive":
            st.success(f"✅ Sentiment: **{prediction}**")
        else:
            st.error(f"❌ Sentiment: **{prediction}**")

        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        st.progress(int(confidence))
