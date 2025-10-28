import cv2
import numpy as np
from .BaseUI import BaseUI
from Config import CONFIG

class MovementUI(BaseUI):
    def draw_ui(self, image, core):
        """Desenha UI específica para Libras"""
        title = f"Tradutor de Libras - Modo: {core.mode.upper()}"
        image = self.draw_base_ui(image, title, core.mode, core.samples_count, core.current_word)
        
        height, width = image.shape[:2]
        scale_factor = self.calculate_scale_factor(width)
        text_color = (255, 255, 255)

        if core.mode == "treino":
            status_text = f"Treinando: {core.new_movement_name}"
            recording_color = (0, 0, 255) if core.is_recording else (255, 255, 0)
            recording_status = "GRAVANDO" if core.is_recording else "PAUSADO"
            
            cv2.putText(
                image,
                f"{status_text} | {recording_status} | Amostras: {core.samples_count}",
                (int(10 * scale_factor), int(100 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * scale_factor,
                recording_color,
                2,
                cv2.LINE_AA,
            )

        # Informações do modelo
        model_info = core.get_model_info()
        cv2.putText(
            image,
            model_info,
            (int(width - 400 * scale_factor), int(35 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale_factor,
            (200, 200, 255),
            1,
            cv2.LINE_AA,
        )

        # Predição atual (modo teste) - CORREÇÃO APLICADA (opcional): Adicionar tipo estático/dinâmico
        if core.mode == "teste" and core.current_prediction:
            confidence_color = (0, 255, 0) if core.prediction_confidence > 0.8 else (0, 255, 255)
            type_text = 'Dinâmico' if core.is_dynamic else 'Estático'
            prediction_text = f"Sinal Detectado: {core.current_prediction} ({core.prediction_confidence:.2f}) - {type_text}"
            
            cv2.putText(
                image,
                prediction_text,
                (int(width // 2 - 200 * scale_factor), int(100 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7 * scale_factor,
                confidence_color,
                2,
                cv2.LINE_AA,
            )

        # 🔥 INSTRUÇÕES ATUALIZADAS PARA LIBRAS
        if core.mode == "teste":
            instructions = "Q:Sair | C:Limpar | T:Treino | D:Deletar | M:Gestos | R:Gravar"
        else:
            instructions = "Q:Sair | S:Salvar | N:Nome | R:Gravar/Pausar | ESC:Cancelar | M:Gestos"
        
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

        # 🔥 LEGENDA DE CORES PARA MÃOS
        cv2.putText(
            image,
            "Mao Esquerda      Mao Direita",
            (int(width - 300 * scale_factor), int(height - 40 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 * scale_factor,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return image

    def resize_for_display(self, image):
        """Redimensiona imagem mantendo proporção"""
        window_width, window_height = CONFIG["window_size"]
        
        original_height, original_width = image.shape[:2]
        
        # Calcular ratio mantendo proporção
        ratio = min(window_width / original_width, window_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Redimensionar
        resized = cv2.resize(image, (new_width, new_height))
        
        # Adicionar bordas se necessário
        if new_width < window_width or new_height < window_height:
            display_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            x_offset = (window_width - new_width) // 2
            y_offset = (window_height - new_height) // 2
            display_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            return display_image
        
        return resized