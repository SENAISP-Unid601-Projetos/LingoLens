import cv2

class UIManager:
    def __init__(self, ui_scale=1.0):
        self.ui_scale = ui_scale
        self.show_help = False
        self.show_text_input = False
        self.input_text = ""
        self.input_prompt = ""
        self.error_message = ""
        self.error_message_timer = 0
        self.error_message_duration = 90

    def draw_ui(self, image, status, cooldown, word, help_menu=False, error=""):
        height, width = image.shape[:2]
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)

        # Barra de status
        cv2.putText(image, status,
                    (int(10*self.ui_scale), int(35*self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9*self.ui_scale,
                    outline_color, 3, cv2.LINE_AA)
        cv2.putText(image, status,
                    (int(10*self.ui_scale), int(35*self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9*self.ui_scale,
                    text_color, 2, cv2.LINE_AA)

        # Palavra atual
        cv2.putText(image, word,
                    (width//2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2*self.ui_scale,
                    (0, 255, 0), 3, cv2.LINE_AA)

        # Instruções
        instructions = "Q:Sair C:Limpar N:Num/Letra T:Treino S:Gesto E:Exportar H:Ajuda"
        cv2.putText(image, instructions,
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7*self.ui_scale,
                    text_color, 2, cv2.LINE_AA)

        # Erro
        if error:
            cv2.putText(image, error,
                        (10, height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7*self.ui_scale,
                        (255, 50, 50), 2, cv2.LINE_AA)
