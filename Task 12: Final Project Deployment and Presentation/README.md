# 🚢 Titanic Survival Prediction - Final Project

## 📌 Problem Statement
Predict whether a Titanic passenger survived or not based on features like age, gender, passenger class, and fare. This is a binary classification problem solved using a Random Forest Classifier.

---

## 📂 Dataset Description
- Source: Titanic Dataset (Kaggle / DataScienceDojo)
- Total Passengers: 891
- Target Column: Survived (0 = Did Not Survive, 1 = Survived)
- Features Used: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

---

## 🔧 Preprocessing Pipeline
- Dropped irrelevant columns: Name, Ticket, Cabin, PassengerId
- Filled missing Age values with median
- Filled missing Embarked values with mode
- Label encoded categorical columns: Sex, Embarked
- Standard scaled numeric columns: Age, Fare, SibSp, Parch

---

## 📊 Exploratory Data Analysis (6 Charts)
1. Survival Count — Only 38% of passengers survived
2. Survival by Gender — Females had much higher survival rate
3. Age Distribution — Most passengers were aged 20 to 40
4. Survival by Passenger Class — First class had highest survival rate
5. Fare Distribution by Survival — Survivors paid higher fares
6. Feature Correlation Heatmap — Pclass negatively correlated with Fare

---

## 🤖 Model Used
- Algorithm: Random Forest Classifier
- Number of Trees: 200
- Max Depth: 8
- Random State: 42
- Train/Test Split: 80% Training / 20% Testing (Stratified)

---

## 📈 Evaluation Results
| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~82%   |
| ROC-AUC   | ~0.87  |
| Precision | ~80%   |
| Recall    | ~76%   |

---

## 🖥️ Streamlit Web App
The app has 3 pages:
- Home and Predict — Enter passenger details and get survival probability
- EDA Charts — View all 6 exploratory data analysis charts
- Model Evaluation — View accuracy, ROC-AUC and classification report

### App Screenshot
![App Screenshot](screenshot.png)
<img width="1364" height="643" alt="app screenshot" src="https://github.com/user-attachments/assets/ec6ab95c-849b-4ef5-b7ff-69c2eff36d0c" />


---

## 🚀 Setup Instructions

### Step 1 - Clone the Repository
git clone https://github.com/Sidrah-Shaikh/coretech-ai-ml-internship/tree/main/Task%2012%3A%20Final%20Project%20Deployment%20and%20Presentation
cd titanic-final-project

### Step 2 - Install Dependencies
pip install -r requirements.txt

### Step 3 - Run the Streamlit App
streamlit run app.py

### Step 4 - Open in Browser
Open your browser and go to:
http://localhost:8501

---

## 📁 Project Structure
- app.py — Streamlit web application
- preprocess.py — Data preprocessing pipeline
- train_model.py — Model training and evaluation script
- requirements.txt — All Python dependencies
- README.md — Project documentation
- screenshot.png — Screenshot of deployed Streamlit app
- dataset/data.csv — Raw Titanic dataset
- dataset/processed_data.csv — Cleaned and processed dataset
- models/model.pkl — Trained Random Forest model
- models/scaler.pkl — Fitted StandardScaler object
- models/encoders.pkl — Fitted LabelEncoder objects
- models/eval_results.json — Model evaluation metrics
- eda_charts/01_survival_count.png
- 
  <img width="823" height="598" alt="01_survival_count" src="https://github.com/user-attachments/assets/34ecc870-39ce-4bee-9e81-b6c1d44a2f81" />

- eda_charts/02_survival_by_gender.png
  
  <img width="823" height="598" alt="02_survival_by_gender" src="https://github.com/user-attachments/assets/1291b9b4-4ad4-467a-8ed6-510f0d5801b9" />

- eda_charts/03_age_distribution.png

  <img width="1041" height="598" alt="03_age_distribution" src="https://github.com/user-attachments/assets/3168116a-9c16-47bf-89d1-b2b19756f397" />

- eda_charts/04_survival_by_class.png

  <img width="823" height="598" alt="04_survival_by_class" src="https://github.com/user-attachments/assets/ae1f8606-fb8f-4011-9d0f-fde95e050f64" />

- eda_charts/05_fare_by_survival.png

  <img width="1055" height="598" alt="05_fare_by_survival" src="https://github.com/user-attachments/assets/3f8d0744-f80d-47b4-b1e9-c1d3dbfcee3d" />

- eda_charts/06_correlation_heatmap.png

<img width="1088" height="912" alt="06_correlation_heatmap" src="https://github.com/user-attachments/assets/00982f44-b3c0-4335-bb3d-07060c6c63fd" />

---

## 🎬 Demo Video
Watch the full project walkthrough here:
https://www.loom.com/share/f7c5c40456c345a5a261924e882f7ea5

---

## 👤 Author
- Name: Sidra Abdul Shakoor
- Task: Task 12 - Final Project Deployment and Presentation
- Internship: AI/ML Internship
