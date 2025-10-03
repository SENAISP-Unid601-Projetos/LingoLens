import cv2
import time

class UIManager:
    def __init__(self, ui_scale=1.0):
        self.ui_scale = ui_scale
        self.show_help = False
        self.cursor_blink_time = 0

    def draw_ui(self, image, status, cooldown, word, sample_count=0, input_text="", is_input_active=False):
        height, width = image.shape[:2]
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)

        # Barra de status com contador de samples
        status_text = f"{status} | Samples: {sample_count}/100" if "Treino" in status else status
        cv2.putText(image, status_text,
                    (int(10*self.ui_scale), int(35*self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9*self.ui_scale,
                    outline_color, 3, cv2.LINE_AA)
        cv2.putText(image, status_text,
                    (int(10*self.ui_scale), int(35*self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9*self.ui_scale,
                    text_color, 2, cv2.LINE_AA)

        # Palavra atual
        cv2.putText(image, word,
                    (width//2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2*self.ui_scale,
                    (0, 255, 0), 3, cv2.LINE_AA)

        # Caixa de texto
        if is_input_active:
            box_top_left = (10, 80)
            box_bottom_right = (300, 120)
            cv2.rectangle(image, box_top_left, box_bottom_right, (0, 0, 255), 2)
            cursor = "_" if (time.time() - self.cursor_blink_time) % 1 < 0.5 else ""
            display_text = f"Nome do gesto: {input_text}{cursor}"
            cv2.putText(
                image, display_text,
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6*self.ui_scale,
                text_color, 2, cv2.LINE_AA
            )
            cv2.putText(
                image, "Letras A-Z, Enter=Confirmar, Backspace=Apagar",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6*self.ui_scale,
                text_color, 2, cv2.LINE_AA
            )

        # Instruções
        instructions = "Q:Sair C:Limpar T:Treino S:Gesto"
        cv2.putText(image, instructions,
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7*self.ui_scale,
                    text_color, 2, cv2.LINE_AA)

        # Erro
        if cooldown > 0:
            cv2.putText(image, "Aguarde para nova predição...",
                        (10, height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7*self.ui_scale,
                        (255, 50, 50), 2, cv2.LINE_AA)