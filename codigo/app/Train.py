import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def preparar_dados(dados):
    X, y = [], []
    for nome, landmarks_str in dados:
        landmarks = json.loads(landmarks_str)
        vetor = np.array(landmarks).flatten()
        X.append(vetor)
        y.append(nome)
    return np.array(X), np.array(y)

def treinar_modelo(dados, save_path="gesture_model.pkl"):
    X, y = preparar_dados(dados)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y_encoded)

    joblib.dump((model, encoder), save_path)
    return model, encoder
