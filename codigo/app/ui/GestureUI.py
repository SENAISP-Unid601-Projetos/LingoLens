import cv2
from .BaseUI import BaseUI
from Config import CONFIG

class GestureUI(BaseUI):
    def draw_ui(self, image, core):
        """Desenha UI completa para gestos"""
        title = "Reconhecimento de Gestos" if core.mode == "teste" else "Treino de Gestos"
        image = self.draw_base_ui(image, title, core.mode, core.samples_count, core.current_word)
        
        height, width = image.shape[:2]
        scale_factor = self.calculate_scale_factor(width)
        text_color = (255, 255, 255)

        # Instruções específicas
        if core.mode == "teste":
            instructions = "Q:Sair C:Limpar T:Treino D:Deletar M:Movimentos "
        else:
            instructions = "Q:Sair S:Salvar N:Nome ESC:Cancelar M:Movimentos "
        
        cv2.putText(
            image,
            instructions,
            (int(10 * scale_factor), int(height - 20 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale_factor,
            text_color,
            2,
            cv2.LINE_AA,
        )

        return image

    def resize_for_display(self, image):
        """Redimensiona imagem para display mantendo a proporção"""
        window_width, window_height = CONFIG["window_size"]

        # Obter dimensões originais
        original_height, original_width = image.shape[:2]

        # Calcular ratio de redimensionamento mantendo a proporção
        ratio = min(window_width / original_width, window_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # Redimensionar mantendo a proporção
        resized = cv2.resize(image, (new_width, new_height))

        # Se necessário, adicionar bordas para centralizar
        if new_width < window_width or new_height < window_height:
            # Criar imagem preta do tamanho da janela
            display_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)

            # Calcular offsets para centralizar
            x_offset = (window_width - new_width) // 2
            y_offset = (window_height - new_height) // 2

            # Colocar imagem redimensionada no centro
            display_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            return display_image
        else:
            return resized