import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['label'] = dataset.target

    print("First few rows:\n", df.head())
    print("Summary statistics:\n", df.describe())
    print("Label distribution:\n", df['label'].value_counts())

    X = df.drop(columns='label', axis=1)
    Y = df['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std, Y_train, Y_test, scaler, dataset