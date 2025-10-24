import cv2
import time
from Config import CONFIG

class UIManager:
    def __init__(self, ui_scale=1.0):
        self.ui_scale = ui_scale
        self.cursor_blink_time = 0
        self.sequence_buffer = []  # Para barra de progresso
        self.instructions = {
            "letter": "Mostre a letra na câmera e pressione 's' para salvar",
            "word": "Grave a sequência de movimento e pressione 's' para salvar",
            "movement": "Grave a sequência de movimento e pressione 's' para salvar"
        }

    def draw_ui(self, image, status, cooldown, word, sample_count=0, input_text="", is_input_active=False, current_letter="", gesture_list=None, selected_index=None):
        height, width = image.shape[:2]
        text_color = (255, 255, 255)  # Branco
        outline_color = (0, 0, 0)     # Preto
        bg_color = (50, 50, 50)       # Cinza escuro para fundo

        # Fundo semi-transparente para status
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), bg_color, -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Status com contador de samples e tipo de gesto
        gesture_type = status.split('(')[-1].strip(')') if '(' in status else "letter"
        if "Treino" in status:
            status_text = f"Modo: Treino | Tipo: {gesture_type} | Samples: {sample_count}/{CONFIG['min_samples_per_class']}"
        elif gesture_list is not None:
            status_text = f"Modo: Excluir | Tipo: {gesture_type} | Selecione: {selected_index + 1 if gesture_list else 0}/{len(gesture_list)}"
        else:
            status_text = f"Modo: Teste | Tipo: {gesture_type} | Palavra: {word}"
        
        cv2.putText(image, status_text,
                    (int(10 * self.ui_scale), int(35 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * self.ui_scale,
                    outline_color, 3, cv2.LINE_AA)
        cv2.putText(image, status_text,
                    (int(10 * self.ui_scale), int(35 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * self.ui_scale,
                    text_color, 2, cv2.LINE_AA)

        # Gesto atual (se definido)
        if current_letter:
            cv2.putText(image, f"Gesto: {current_letter}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        # Barra de progresso para gestos dinâmicos
        if gesture_type in ["word", "movement"] and hasattr(self, 'sequence_buffer'):
            progress = len(self.sequence_buffer) / CONFIG["max_sequence_length"] * 100
            cv2.rectangle(image, (10, 100), (10 + int(progress * 2), 120), (0, 255, 0), -1)
            cv2.putText(image, f"Frames: {len(self.sequence_buffer)}/{CONFIG['max_sequence_length']}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.ui_scale, text_color, 2)

        # Exibir lista de gestos para exclusão
        if gesture_list is not None:
            # Calcular altura do retângulo com base no número de gestos
            item_height = 30  # Altura de cada item
            padding = 20      # Espaçamento extra no topo e na base
            list_height = item_height * len(gesture_list) + padding * 3
            list_width = 450  # Aumentado para acomodar textos mais longos
            list_top_left = (10, 100)
            list_bottom_right = (10 + list_width, 100 + list_height)

            # Fundo semi-transparente para a lista
            overlay = image.copy()
            cv2.rectangle(overlay, list_top_left, list_bottom_right, (30, 30, 30), -1)  # Cinza mais escuro
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # Título da lista
            cv2.putText(image, "SELECIONE O GESTO PARA EXCLUIR:",
                        (15, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                        (255, 255, 0), 2, cv2.LINE_AA)  # Amarelo para destaque

            # Itens da lista
            for i, gesture in enumerate(gesture_list):
                y_position = 150 + i * item_height  # Aumentado o espaçamento vertical
                if i == selected_index:
                    # Fundo verde para item selecionado
                    cv2.rectangle(image, (15, y_position - 15), (15 + list_width - 10, y_position + 10),
                                (0, 100, 0), -1)  # Verde escuro
                    cv2.putText(image, f"> {gesture}",
                                (20, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                                text_color, 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, f"  {gesture}",
                                (20, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                                text_color, 2, cv2.LINE_AA)

            # Instruções de navegação
            cv2.putText(image, "Setas ou N/P: Navegar | ENTER: Excluir | ESC: Sair",
                        (15, 130 + len(gesture_list) * item_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.ui_scale,
                        (0, 255, 255), 2, cv2.LINE_AA)  # Ciano para destaque

        # Caixa de texto para entrada de novo gesto
        elif is_input_active:
            box_top_left = (10, 130)
            box_bottom_right = (300, 170)
            cv2.rectangle(image, box_top_left, box_bottom_right, (0, 0, 255), 2)
            cursor = "_" if (time.time() - self.cursor_blink_time) % 1 < 0.5 else ""
            display_text = f"Nome do gesto: {input_text}{cursor}"
            cv2.putText(
                image, display_text,
                (15, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.ui_scale,
                text_color, 2, cv2.LINE_AA
            )
            cv2.putText(
                image, "Letras A-Z, Enter=Confirmar, Backspace=Apagar",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.ui_scale,
                text_color, 2, cv2.LINE_AA
            )
        elif "Treino" in status:
            cv2.putText(image, self.instructions.get(gesture_type, "Mostre o gesto na câmera"),
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.ui_scale,
                        text_color, 2, cv2.LINE_AA)

        # Instruções gerais
        instructions = "Q:Sair C:Limpar T:Treino S:Gesto D:Excluir"
        cv2.putText(image, instructions,
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                    text_color, 2, cv2.LINE_AA)

        # Mensagem de cooldown
        if cooldown > 0:
            cv2.putText(image, "Aguarde para nova predicao...",
                        (10, height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                        (255, 50, 50), 2, cv2.LINE_AA)

        return image