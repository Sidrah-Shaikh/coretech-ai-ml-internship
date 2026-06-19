
import streamlit as st

st.title("Client Feedback Sentiment Analysis")

feedback = st.text_input("Enter Feedback")

if feedback:
    st.success("Feedback received")
