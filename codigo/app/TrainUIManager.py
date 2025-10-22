import cv2


class TrainUIManager:
    def __init__(self, ui_scale=1.0):
        self.ui_scale = ui_scale
        self.current_letter = ""
        self.show_text_input = False
        self.input_text = ""
        self.input_prompt = ""
        self.error_message = ""
        self.error_message_timer = 0
        self.error_message_duration = 90
        self.input_action = ""

    def draw_train_ui(self, image, status="Treino", word=""):
        """
        Desenha a interface para treino de gestos/letras igual à de movimentos.
        """
        height, width = image.shape[:2]
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)

        # Título da tela
        title = f"Treinar Gestos - Letra: {self.current_letter}"
        cv2.putText(
            image,
            title,
            (int(10 * self.ui_scale), int(35 * self.ui_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0 * self.ui_scale,
            outline_color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            title,
            (int(10 * self.ui_scale), int(35 * self.ui_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0 * self.ui_scale,
            text_color,
            2,
            cv2.LINE_AA,
        )

        # Status/Modo
        cv2.putText(
            image,
            status,
            (int(10 * self.ui_scale), int(70 * self.ui_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * self.ui_scale,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Palavra atual (se aplicável)
        if word:
            cv2.putText(
                image,
                word,
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * self.ui_scale,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        # Instruções
        instructions = "S:Salvar Gesto | Q:Sair | C:Limpar | M:Voltar para Movimentos"
        cv2.putText(
            image,
            instructions,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 * self.ui_scale,
            text_color,
            2,
            cv2.LINE_AA,
        )

        # Mensagem de erro
        if self.error_message:
            cv2.putText(
                image,
                self.error_message,
                (10, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7 * self.ui_scale,
                (255, 50, 50),
                2,
                cv2.LINE_AA,
            )

        # Input de texto (se necessário)
        if self.show_text_input:
            # Fundo para input
            overlay = image.copy()
            cv2.rectangle(overlay, (50, 100), (width - 50, 200), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Prompt
            cv2.putText(
                image,
                self.input_prompt,
                (70, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            
            # Texto digitado
            cv2.putText(
                image,
                self.input_text + "_",
                (70, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        return image

    def set_current_letter(self, letter):
        self.current_letter = letter.upper()

    def set_error(self, text):
        self.error_message = text
        self.error_message_timer = 0