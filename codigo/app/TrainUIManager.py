import cv2
from Config import CONFIG

class TrainUIManager:
    def __init__(self):
        self.current_letter = ""
        self.instruction = "Mostre a letra na câmera e pressione 's' para salvar"

    def draw_train_ui(self, frame):
        """
        Desenha a interface para treino de gestos/letras.
        """
        h, w, _ = frame.shape

        # Fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Texto de instrução
        cv2.putText(frame, self.instruction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Letra atual
        cv2.putText(frame, f"Letra: {self.current_letter}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        return frame

    def set_current_letter(self, letter):
        self.current_letter = letter.upper()
