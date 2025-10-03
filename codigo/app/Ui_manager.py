import cv2

class UIManager:
    def __init__(self, ui_scale=1.0):
        self.ui_scale = ui_scale
        self.show_help = False
        self.show_text_input = False
        self.input_text = ""
        self.input_prompt = ""
        self.error_message = ""

    def draw_ui(self, image, status, cooldown, word):
        height, width = image.shape[:2]
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)

        # Barra de status
        cv2.putText(image, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, outline_color, 3, cv2.LINE_AA)
        cv2.putText(image, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

        # Palavra atual
        cv2.putText(image, word, (width//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # Instruções
        instructions = "Q:Sair C:Limpar T:Treino S:Salvar N:Nome H:Ajuda"
        cv2.putText(image, instructions, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

        # Mensagem de erro
        if self.error_message:
            cv2.putText(image, self.error_message, (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2, cv2.LINE_AA)

        # Input de texto
        if self.show_text_input:
            prompt = self.input_prompt + " " + self.input_text
            cv2.putText(image, prompt, (10, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 50), 2, cv2.LINE_AA)

        # Ajuda
        if self.show_help:
            help_text = ["Use a câmera para mostrar gestos.", "Modo treino: pressione 'T', defina nome 'N', mova a mão e 'S' para salvar."]
            y0 = 100
            for i, line in enumerate(help_text):
                cv2.putText(image, line, (10, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        return image

    def set_error(self, message):
        self.error_message = message
