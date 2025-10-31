import cv2
import numpy as np
from .BaseUI import BaseUI
from Config import CONFIG

class LibrasUI(BaseUI):
    def draw_ui(self, image, core):
        """Desenha UI completa para Tradutor de Libras"""
        title = f"Tradutor de Libras - {core.recognition_mode.upper()}"
        image = self.draw_base_ui(image, title, core.mode, core.samples_count, core.current_word)
        
        height, width = image.shape[:2]
        scale_factor = self.calculate_scale_factor(width)

        # 🔥 INFORMAÇÕES DE MODO E ESTADO
        mode_info = self._get_mode_info(core)
        cv2.putText(
            image,
            mode_info,
            (int(width - 350 * scale_factor), int(70 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale_factor,
            (200, 200, 255),
            1,
            cv2.LINE_AA,
        )

        # 🔥 ESTADO DE TREINO
        if core.mode == "treino":
            training_info = f"Treinando: {core.new_gesture_name} | Tipo: {core.gesture_type} | Amostras: {core.samples_count}"
            cv2.putText(
                image,
                training_info,
                (int(10 * scale_factor), int(100 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * scale_factor,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 🔥 LEGENDA DE TECLAS
        instructions = self._get_instructions(core.mode)
        cv2.putText(
            image,
            instructions,
            (int(10 * scale_factor), int(height - 20 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 * scale_factor,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return image

    def _get_mode_info(self, core):
        """Retorna informações sobre o modo atual"""
        if core.recognition_mode == "letras":
            return "📝 Soletração: Reconhece letras A-Z"
        elif core.recognition_mode == "palavras":
            return "🔤 Palavras: Reconhece sinais completos"
        elif core.recognition_mode == "frases":
            return "💬 Frases: Reconhece sinais complexos"
        return ""

    def _get_instructions(self, mode):
        """Retorna instruções baseadas no modo"""
        if mode == "teste":
            return "Q:Sair | C:Limpar | M:Mudar Modo | T:Treino | D:Deletar | R:Reset | +/-:Resolução"
        else:
            return "Q:Sair | S:Salvar | N:Nome | ESC:Cancelar | T:Tipo"

    def resize_for_display(self, image):
        """Redimensiona imagem para display mantendo proporção"""
        window_width, window_height = CONFIG["window_size"]
        original_height, original_width = image.shape[:2]

        ratio = min(window_width / original_width, window_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        resized = cv2.resize(image, (new_width, new_height))

        if new_width < window_width or new_height < window_height:
            display_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            x_offset = (window_width - new_width) // 2
            y_offset = (window_height - new_height) // 2
            display_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            return display_image
        
        return resized