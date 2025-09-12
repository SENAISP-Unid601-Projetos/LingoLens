import numpy as np
import pickle

def carregar_modelo(model_path="modelo.pkl", encoder_path="encoder.pkl"):
    """
    Carrega modelo e encoder do disco.
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        model = None

    try:
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
    except FileNotFoundError:
        encoder = None

    return model, encoder


def extrair_landmarks(hand_landmarks):
    """
    Converte NormalizedLandmarkList do MediaPipe
    em vetor 1D de floats compatível com sklearn.
    """
    if hand_landmarks is None:
        return None
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()


def prever_gesto(model, encoder, hand_landmarks):
    """
    Recebe o objeto landmarks do MediaPipe, transforma e prevê gesto.
    """
    vetor = extrair_landmarks(hand_landmarks)
    if vetor is None:
        return "Nenhum gesto"
    
    vetor = vetor.reshape(1, -1)  # sklearn espera shape (1, n_features)
    pred = model.predict(vetor)

    if encoder:
        return encoder.inverse_transform(pred)[0]
    return pred[0]
