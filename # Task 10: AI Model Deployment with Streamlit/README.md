# Task 10: AI Model Deployment with Streamlit

## Objective
The objective of this project is to deploy a Machine Learning model using Streamlit and create an interactive web application that predicts whether a student will pass or fail based on input features.

---

## Dataset
**Student Pass Prediction Dataset**

The dataset contains student-related information used to predict academic performance.

---

## Technologies and Libraries Used

- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Streamlit
- Matplotlib

---

## Project Workflow

### 1. Data Preparation
- Created dataset
- Loaded dataset
- Checked dataset structure
- Verified data quality

### 2. Exploratory Data Analysis (EDA)
- Examined dataset statistics
- Visualized data patterns
- Analyzed feature relationships

### 3. Model Training
- Split dataset into training and testing sets
- Trained Random Forest Classifier
- Generated predictions

### 4. Model Evaluation
- Calculated model accuracy
- Generated Confusion Matrix
- Evaluated model performance

### 5. Model Saving
- Saved trained model using Joblib

```python
joblib.dump(model, "model.pkl")
```

### 6. Streamlit Deployment
- Created interactive web application
- Added user input fields
- Implemented prediction button
- Displayed prediction results

---

## Project Structure

```text
Task10_AI_Model_Deployment/
│
├── Task_10.ipynb
├── student_data.csv
├── model.pkl
├── app.py
├── requirements.txt
├── confusion_matrix.png
├── README.md
```

---

## How to Run the Application

### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## Sample Input

| Study Hours | Attendance |
|------------|------------|
| 7 | 85 |

---

## Sample Output

```text
Prediction: Pass
```

---

## Results

- Successfully trained a Random Forest Classifier.
- Evaluated model performance using a Confusion Matrix.
- Saved the trained model using Joblib.
- Developed an interactive Streamlit web application.
- Successfully deployed a machine learning prediction system.
- Enabled users to enter input values and receive real-time predictions.

---

## Future Improvements

- Deploy application on Streamlit Cloud.
- Add additional student performance features.
- Improve model accuracy using Hyperparameter Tuning.
- Enhance user interface and user experience.
- Add graphical visualizations to the web application.

---

## Conclusion

This project demonstrates the complete deployment cycle of a Machine Learning model using Streamlit. The trained Random Forest model was successfully saved, integrated into a Streamlit application, and used to generate predictions through an interactive user interface.
