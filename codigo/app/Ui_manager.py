import cv2
import numpy as np
import os

class UIManager:
    def __init__(self):
        """Inicializa o gerenciador de interface do usuário com UI responsiva."""
        self.instructions = "Q=Sair |C=Limpar |T=Treino Estático |M=Treino Dinâmico |R=Reconhecimento |S=Salvar |D=Excluir"
        self.line_type = cv2.LINE_AA

    def _get_scaled_font(self, width, base_scale=0.5):
        """Ajusta escala da fonte com base na largura da tela."""
        scale = max(0.4, base_scale * (width / 800))
        thickness = max(1, int(scale * 2))
        return scale, thickness

    def draw_ui(self, image, status, prediction_cooldown, current_word, sample_count, input_text, 
                is_input_active, new_gesture_name, gesture_list=None, selected_index=None, 
                hand_stable=False, variance=0.0, dynamic_status="", mode="", 
                dynamic_sequence_length=0, saving_status=""):
        """Desenha UI responsiva que se adapta a qualquer tamanho de janela."""
        try:
            if image is None or image.size == 0:
                return image

            height, width = image.shape[:2]
            if width == 0 or height == 0:
                return image

            font_scale, thickness = self._get_scaled_font(width)
            margin = max(10, int(width * 0.02))
            line_height = max(20, int(font_scale * 40))

            # Fundo escuro na parte inferior
            bar_height = line_height * 3
            overlay = image.copy()
            cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            def draw_text(text, y, color=(200, 200, 200), align='left', bold=False):
                scale = min(font_scale * (1.3 if bold else 1.0), 1.5)  # Limite máximo
                thick = min(thickness + (1 if bold else 0), 4)
                size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
                if align == 'center':
                    x = (width - size[0]) // 2
                elif align == 'right':
                    x = width - size[0] - margin
                else:
                    x = margin
                cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, self.line_type)

            # Linha 1: Status principal
            draw_text(status, margin + line_height * 0, (0, 255, 255), bold=True)

            # Linha 2: Modo específico
            y2 = margin + line_height * 1
            if is_input_active:
                draw_text(f"Nome: {input_text}_", y2, (0, 255, 0))
            elif mode == "train_static" and new_gesture_name:
                stability = "ESTÁVEL" if hand_stable else "INSTÁVEL"
                color = (0, 255, 0) if hand_stable else (0, 0, 255)
                draw_text(f"Treinando: {new_gesture_name} ({sample_count} amostras, {stability})", y2, color)
            elif mode == "train_dynamic" and new_gesture_name:
                draw_text(f"Capturando: {new_gesture_name} ({dynamic_sequence_length} frames)", y2, (255, 255, 0))
            elif mode == "recognize":
                if current_word:
                    draw_text(f"Letra: {current_word}", y2, (0, 255, 0))
                if dynamic_status:
                    draw_text(dynamic_status, y2 + line_height, (255, 255, 0))

            # Status temporário
            if saving_status:
                draw_text(saving_status, height - bar_height - margin, (0, 255, 255), bold=True)

            # Modo exclusão
            if gesture_list:
                start_y = margin
                for i, gesture in enumerate(gesture_list[:7]):
                    color = (0, 255, 0) if i == selected_index else (180, 180, 180)
                    draw_text(gesture, start_y + i * line_height, color, align='right')

            # Instruções
            draw_text(self.instructions, height - margin, (180, 180, 180), align='center')

            return image
        except Exception as e:
            print(f"[ERROR] Erro ao desenhar UI: {e}")
            return image