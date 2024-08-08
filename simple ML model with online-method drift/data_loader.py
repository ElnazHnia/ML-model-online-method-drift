import pandas as pd
import numpy as np
import torch

def load_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    data = pd.read_csv(url)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    data = data[features + ['Survived']]
    fill_values = {'Age': data['Age'].median(), 'Fare': data['Fare'].median(), 'Embarked': 'S'}
    data.fillna(value=fill_values, inplace=True)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return data, features


def prepare_tensors(data, features):
    X = data[features].values
    stds = X.std(axis=0)
    means = X.mean(axis=0)
    X_normalized = (X - means) / (stds + 1e-8)  # Adding epsilon to avoid divide by zero
    if np.any(np.isnan(X_normalized)) or np.any(np.isinf(X_normalized)):
        print("NaNs or infinite values found in normalized data.")

    y = data['Survived'].values
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor
