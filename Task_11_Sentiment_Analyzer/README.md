#  Client Feedback Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Sklearn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)

A machine learning web application that analyzes customer feedback and predicts whether the sentiment is **Positive** or **Negative** along with a confidence score.

---

##  Project Structure

Task_11_Sentiment_Analyzer/
- Task_11_Sentiment_Analyzer.ipynb — Main notebook
- app.py — Streamlit web app
- model.pkl — Trained ML model
- vectorizer.pkl — TF-IDF vectorizer
- model_comparison.csv — Model accuracy results
- sentiment_distribution.png — EDA chart  
- text_length_distribution.png — EDA chart
- wordcloud.png — Word cloud visualization
- model_comparison.png — Model comparison chart
- confusion_matrix.png — Confusion matrix
- README.md — Project documentation

---

##  Project Overview

This project is part of **Task 11 — Final AI/ML Project Development**.

The goal is to build an end-to-end sentiment analysis system that:
- Takes real customer/Twitter feedback as input
- Preprocesses and cleans the text
- Trains and compares multiple ML models
- Deploys a working web app for live predictions

---

##  Dataset

- **Source:** Twitter Sentiment Analysis Dataset
- **Size:** 5,000 balanced samples (2,500 Positive + 2,500 Negative)
- **Columns Used:** Feedback, Sentiment
- **Labels:** Positive / Negative

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data handling |
| Matplotlib & Seaborn | Visualizations |
| WordCloud | Text visualization |
| Scikit-learn | ML models & evaluation |
| Streamlit | Web app deployment |
| Joblib | Model saving/loading |
| Google Colab | Development environment |

---

##  Steps Performed

### 1. Data Loading
- Loaded real Twitter sentiment dataset from URL
- Mapped labels: 0 → Negative, 1 → Positive
- Balanced dataset with equal class samples

### 2. Text Preprocessing
- Lowercasing
- Removing URLs, mentions (@user), hashtags
- Removing numbers and punctuation
- Stripping extra whitespace

### 3. Exploratory Data Analysis (EDA)
- Sentiment distribution countplot
- Text length distribution by sentiment
- Word clouds for Positive and Negative feedback

### 4. Feature Extraction
- TF-IDF Vectorization (max_features=5000, ngram_range=(1,2))
- 80/20 Train/Test split with stratification

### 5. Model Training & Comparison

| Model | Accuracy |
|-------|----------|
| Logistic Regression | your result here |
| Naive Bayes | your result here |

### 6. Model Saving
- Best model saved as model.pkl
- Vectorizer saved as vectorizer.pkl

### 7. Streamlit App
- User enters feedback text
- App cleans text and predicts sentiment
- Displays Positive/Negative label + confidence score

---

##  Business Insights

1. Logistic Regression outperformed Naive Bayes on this dataset
2. Positive and Negative feedback is balanced — no class bias
3. Negative reviews contain words like bad, terrible, poor — useful for flagging service issues
4. Automating sentiment analysis saves manual review time significantly
5. Confidence scores help prioritize which feedback needs urgent attention
6. This system can be integrated into CRM tools for real-time feedback monitoring

---

##  Author

**Sidra Abdul Shakoor**
AI/ML Student — Task 11 Final Project

---

