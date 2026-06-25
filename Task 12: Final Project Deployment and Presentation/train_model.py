import pandas as pd
import pickle, json, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from preprocess import load_data, preprocess

os.makedirs('models', exist_ok=True)
df = load_data()
X = df.drop(columns=['Survived'])
y = df['Survived']
X_processed, scaler, encoders = preprocess(X, fit=True)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {acc:.4f}')
print(f'ROC-AUC: {auc:.4f}')
pickle.dump(model, open('models/model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(encoders, open('models/encoders.pkl', 'wb'))
results = {'accuracy': round(float(acc),4), 'roc_auc': round(float(auc),4), 'report': report}
json.dump(results, open('models/eval_results.json','w'), indent=2)
print('Model saved!')
