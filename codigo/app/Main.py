import cv2
from Capture import capturar_landmarks, desenhar_landmarks
from Storage import carregar_dados
from Train import treinar_modelo
from Predict import carregar_modelo, prever_gesto

# === Carregar modelo treinado ou treinar se não existir ===
model, encoder = carregar_modelo()
if model is None:
    print("Treinando modelo...")
    dados = carregar_dados()
    model, encoder = treinar_modelo(dados)

# === Inicializar webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: nao foi possivel acessar a webcam.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = capturar_landmarks(frame)
    if landmarks:
        desenhar_landmarks(frame, landmarks)
        gesto = prever_gesto(model, encoder, landmarks)
        cv2.putText(frame, f"Gesto: {gesto}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Tradutor Libras", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
