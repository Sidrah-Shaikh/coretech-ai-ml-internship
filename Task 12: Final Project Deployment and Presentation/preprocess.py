import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def load_data(path='dataset/data.csv'):
    return pd.read_csv(path)

def preprocess(df, fit=True, scaler=None, encoders=None):
    df = df.copy()
    df.drop(columns=['Name','Ticket','Cabin','PassengerId'], errors='ignore', inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    cat_cols = ['Sex', 'Embarked']
    if encoders is None:
        encoders = {}
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            df[col] = encoders[col].transform(df[col].astype(str))
    num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler, encoders
