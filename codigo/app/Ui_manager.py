import cv2
import numpy as np

class UIManager:
    def __init__(self):
        """Inicializa o gerenciador de interface do usuário."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6  # Reduzido para caber mais texto
        self.color = (255, 255, 255)  # Branco para texto
        self.thickness = 1  # Reduzido para melhor legibilidade
        self.line_type = cv2.LINE_AA
        self.instructions = "Q=Sair, C=Limpar, T=Treino Estático ou Treinar Dinâmico, M=Treino Dinâmico, R=Reconhecimento, S=Salvar, D=Excluir"

    def draw_ui(self, image, status, prediction_cooldown, current_word, sample_count, input_text, 
                is_input_active, new_gesture_name, gesture_list=None, selected_index=None, 
                hand_stable=False, variance=0.0, dynamic_status="", mode="", 
                dynamic_sequence_length=0, saving_status=""):
        """Desenha a interface do usuário na imagem."""
        try:
            # Desenhar status principal
            cv2.putText(image, status, (10, 30), self.font, self.font_scale, self.color, self.thickness, self.line_type)

            # Modo de entrada de texto
            if is_input_active:
                cv2.putText(image, f"Digite o nome do gesto: {input_text}", 
                            (10, 60), self.font, self.font_scale, (0, 255, 0), self.thickness, self.line_type)

            # Modo de treino estático
            elif mode == "train_static" and new_gesture_name:
                stability_text = "Estável" if hand_stable else "Instável"
                color = (0, 255, 0) if hand_stable else (0, 0, 255)
                cv2.putText(image, f"Treinando: {new_gesture_name} ({sample_count} samples, {stability_text}, var={variance:.6f})", 
                            (10, 60), self.font, self.font_scale, color, self.thickness, self.line_type)

            # Modo de treino dinâmico
            elif mode == "train_dynamic" and new_gesture_name:
                cv2.putText(image, f"Treinando dinâmico: {new_gesture_name} ({dynamic_sequence_length} frames)", 
                            (10, 60), self.font, self.font_scale, (255, 255, 0), self.thickness, self.line_type)
                if dynamic_status:
                    cv2.putText(image, dynamic_status, (10, 90), self.font, self.font_scale, (255, 255, 0), self.thickness, self.line_type)

            # Modo de reconhecimento
            elif mode == "recognize":
                if current_word:
                    cv2.putText(image, f"Palavra: {current_word}", 
                                (10, 90), self.font, self.font_scale, self.color, self.thickness, self.line_type)
                if dynamic_status:
                    cv2.putText(image, dynamic_status, (10, 120), self.font, self.font_scale, (255, 255, 0), self.thickness, self.line_type)

            # Modo de exclusão
            if gesture_list:
                cv2.putText(image, "Modo Excluir: Use N/P para navegar, ENTER para excluir, ESC para sair", 
                            (10, image.shape[0] - 60), self.font, self.font_scale, (0, 0, 255), self.thickness, self.line_type)
                for i, gesture in enumerate(gesture_list):
                    color = (0, 255, 0) if i == selected_index else self.color
                    cv2.putText(image, gesture, (image.shape[1] - 200, 30 + i * 30), 
                                self.font, self.font_scale, color, self.thickness, self.line_type)

            # Exibir status de salvamento/treinamento
            if saving_status:
                cv2.putText(image, saving_status, (10, image.shape[0] - 30), 
                            self.font, self.font_scale, (0, 255, 255), self.thickness, self.line_type)

            # Exibir instruções de teclas
            cv2.putText(image, self.instructions, (10, image.shape[0] - 10), 
                        self.font, self.font_scale, (200, 200, 200), self.thickness, self.line_type)

            return image
        except Exception as e:
            print(f"[ERROR] Erro ao desenhar UI: {e}")
            return image