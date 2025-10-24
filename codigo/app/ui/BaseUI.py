import cv2
from Config import CONFIG

class BaseUI:
    def __init__(self):
        self.show_text_input = False
        self.input_text = ""
        self.input_prompt = ""
        self.error_message = ""
        self.error_message_timer = 0
        self.error_message_duration = 90
        self.input_action = ""

    def calculate_scale_factor(self, width):
        """Calcula fator de escala baseado na largura"""
        return width / 640.0

    def draw_base_ui(self, image, title, mode, samples_count, current_word):
        """Desenha elementos básicos da UI"""
        height, width = image.shape[:2]
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)
        scale_factor = self.calculate_scale_factor(width)

        # Título
        cv2.putText(
            image,
            title,
            (int(10 * scale_factor), int(35 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * scale_factor,
            outline_color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            title,
            (int(10 * scale_factor), int(35 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * scale_factor,
            text_color,
            2,
            cv2.LINE_AA,
        )

        # Modo e contador
        if mode == "treino":
            mode_text = f"Modo: TREINO | Amostras: {samples_count}"
        else:
            mode_text = f"Modo: TESTE"
            
        cv2.putText(
            image,
            mode_text,
            (int(10 * scale_factor), int(70 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * scale_factor,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Palavra atual
        cv2.putText(
            image,
            current_word,
            (int(width // 2 - 100 * scale_factor), int(50 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0 * scale_factor,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        # Mensagem de erro
        if self.error_message:
            cv2.putText(
                image,
                self.error_message,
                (int(10 * scale_factor), int(height - 60 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * scale_factor,
                (255, 50, 50),
                2,
                cv2.LINE_AA,
            )

        # Input de texto
        if self.show_text_input:
            self._draw_text_input(image, scale_factor)

        return image

    def _draw_text_input(self, image, scale_factor):
        """Desenha campo de input de texto"""
        height, width = image.shape[:2]
        
        overlay = image.copy()
        input_width = int(400 * scale_factor)
        input_height = int(100 * scale_factor)
        
        cv2.rectangle(overlay, 
                     (int(50 * scale_factor), int(100 * scale_factor)), 
                     (int(50 * scale_factor) + input_width, int(100 * scale_factor) + input_height), 
                     (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        cv2.putText(
            image,
            self.input_prompt,
            (int(70 * scale_factor), int(130 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * scale_factor,
            (255, 255, 255),
            2,
        )
        
        cv2.putText(
            image,
            self.input_text + "_",
            (int(70 * scale_factor), int(170 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 * scale_factor,
            (0, 255, 255),
            2,
        )

    def set_error(self, text):
        self.error_message = text
        self.error_message_timer = 0

    def update_error_timer(self):
        """Atualiza timer da mensagem de erro"""
        if self.error_message_timer > 0:
            self.error_message_timer -= 1
        else:
            self.error_message = ""