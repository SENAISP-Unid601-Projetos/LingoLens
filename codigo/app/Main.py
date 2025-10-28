import cv2
import numpy as np
from core.GestureCore import GestureCore
from core.MovementCore import MovementCore
from ui.GestureUI import GestureUI
from ui.MovementUI import MovementUI
from Database_manager import DatabaseManager
from Config import CONFIG


class GestureApp:
    def __init__(self, db):
        self.core = GestureCore(db)
        self.ui = GestureUI()
        self._should_switch_to_movement = False

    def run(self):
        print("[INFO] Teclas: Q=Sair | C=Limpar | T=Treino | S=Salvar | N=Nome | D=Deletar | M:Movimentos | +/-:Resolução")

        # 🔥 OBTER TAMANHO DA TELA PARA CENTRALIZAR
        window_width, window_height = CONFIG["window_size"]
        
        while True:
            try:
                ret, frame = self.core.cap.read()
                if not ret:
                    self.ui.set_error("Falha ao capturar frame da câmera!")
                    break

                # Processa frame (lógica)
                image, landmarks_list = self.core.process_frame(frame)
                
                # Aplica lógica de negócio
                for landmarks in landmarks_list:
                    if self.core.mode == "teste":
                        result = self.core.predict_gesture(landmarks)
                        if result:
                            self.ui.set_error(result)
                    elif self.core.mode == "treino":
                        self.core.add_training_sample(landmarks)

                # Desenha UI
                image = self.ui.draw_ui(image, self.core)
                display_image = self.ui.resize_for_display(image)
                
                # 🔥 CRIAR JANELA REDIMENSIONÁVEL E CENTRALIZAR
                cv2.namedWindow("GestureApp", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("GestureApp", window_width, window_height)
                cv2.moveWindow("GestureApp", 100, 50)
                
                cv2.imshow("GestureApp", display_image)

                # Processa input
                key = cv2.waitKey(1) & 0xFF
                should_continue = self._handle_input(key)
                if not should_continue:
                    break

                # Atualiza UI
                self.ui.update_error_timer()
                
            except Exception as e:
                print(f"ERRO: {e}")
                import traceback
                traceback.print_exc()
                self.ui.set_error(f"Erro: {str(e)}")

        self.core.cleanup()
        return "movement" if self._should_switch_to_movement else None

    def _handle_input(self, key):
        if self.ui.show_text_input:
            return self._handle_text_input(key)
        else:
            return self._handle_normal_input(key)

    def _handle_text_input(self, key):
        if key == 13 or key == 10:  # Enter
            gesture_name = self.ui.input_text.upper()
            self.ui.input_text = ""
            self.ui.show_text_input = False
            
            if self.ui.input_action == "delete":
                result = self.core.delete_gesture(gesture_name)
                self.ui.set_error(result)
                self.ui.input_action = ""
            else:
                self.core.new_gesture_name = gesture_name
                self.ui.set_error(f"Gesto '{self.core.new_gesture_name}' definido. Pressione 'S' para salvar.")
                
        elif key == 8:  # Backspace
            self.ui.input_text = self.ui.input_text[:-1]
        elif key == 27:  # ESC
            self.ui.show_text_input = False
            self.ui.input_text = ""
            self.ui.input_action = ""
        elif key != 255 and key != 0 and 32 <= key <= 126:
            self.ui.input_text += chr(key)
        
        return True

    def _handle_normal_input(self, key):
        if key == 27:  # ESC
            if self.core.mode == "treino":
                self.core.cancel_training()
                self.ui.set_error("Modo treino cancelado.")

        elif key == ord("q"):
            return False
            
        elif key == ord("m"):
            self._should_switch_to_movement = True
            return False
            
        elif key == ord("c"):
            self.core.current_word = ""
            self.ui.set_error("Texto limpo!")
            
        elif key == ord("t"):
            self.core.start_training_mode()
            self.ui.set_error("Modo Treino ativado. Pressione 'N' para definir nome do gesto.")
            
        elif key == ord("n") and self.core.mode == "treino":
            self.ui.show_text_input = True
            self.ui.input_prompt = "Digite o nome do gesto:"
            self.ui.input_action = ""
            
        elif key == ord("s") and self.core.mode == "treino":
            result = self.core.save_gesture()
            self.ui.set_error(result)
            
        elif key == ord("d") and self.core.mode == "teste":
            self.ui.show_text_input = True
            self.ui.input_prompt = "Digite o nome do gesto para deletar:"
            self.ui.input_action = "delete"
            
        # 🔥 CONTROLES DE RESOLUÇÃO
        elif key == ord("+") or key == ord("="):  # Aumentar resolução
            self._increase_resolution()
        elif key == ord("-") or key == ord("_"):  # Diminuir resolução
            self._decrease_resolution()

        return True

    # 🔥 MÉTODOS PARA CONTROLE DE RESOLUÇÃO
    def _increase_resolution(self):
        """Aumenta a resolução da câmera"""
        current_index = CONFIG["current_resolution_index"]
        if current_index > 0:
            new_index = current_index - 1
            new_width, new_height = CONFIG["resolution_options"][new_index]
            
            if self.core.change_resolution(new_width, new_height):
                CONFIG["current_resolution_index"] = new_index
                CONFIG["camera_resolution"] = (new_width, new_height)
                self.ui.set_error(f"Resolução aumentada para {new_width}x{new_height}")
            else:
                self.ui.set_error("Erro ao aumentar resolução")

    def _decrease_resolution(self):
        """Diminui a resolução da câmera"""
        current_index = CONFIG["current_resolution_index"]
        if current_index < len(CONFIG["resolution_options"]) - 1:
            new_index = current_index + 1
            new_width, new_height = CONFIG["resolution_options"][new_index]
            
            if self.core.change_resolution(new_width, new_height):
                CONFIG["current_resolution_index"] = new_index
                CONFIG["camera_resolution"] = (new_width, new_height)
                self.ui.set_error(f"Resolução diminuída para {new_width}x{new_height}")
            else:
                self.ui.set_error("Erro ao diminuir resolução")


class MovementApp:
    def __init__(self, db):
        self.core = MovementCore(db)
        self.ui = MovementUI()
        self._should_switch_to_gesture = False

    def run(self):
        print("[INFO] Teclas: Q=Sair | C=Limpar | T:Treino | S:Salvar | N=Nome | D:Deletar | R:Gravar | ESC=Cancelar | M:Gestos | +/-:Resolução")
        
        window_width, window_height = CONFIG["window_size"]
        
        while True:
            try:
                ret, frame = self.core.cap.read()
                if not ret:
                    break
                    
                image, landmarks_list = self.core.process_frame(frame)
                
                if self.core.mode == "teste":
                    result = self.core.predict_movement(landmarks_list)
                    if result:
                        self.ui.set_error(result)
                elif self.core.mode == "treino" and self.core.is_recording:
                    pass

                image = self.ui.draw_ui(image, self.core)
                display_image = self.ui.resize_for_display(image)
                
                cv2.namedWindow("MovementApp", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("MovementApp", window_width, window_height)
                cv2.moveWindow("MovementApp", 100, 50)
                
                cv2.imshow("MovementApp", display_image)

                key = cv2.waitKey(1) & 0xFF
                should_continue = self._handle_input(key)
                if not should_continue:
                    break

                self.ui.update_error_timer()
                
            except Exception as e:
                print(f"ERRO: {e}")
                import traceback
                traceback.print_exc()
                self.ui.set_error(f"Erro: {str(e)}")

        self.core.cleanup()
        return "gesture" if self._should_switch_to_gesture else None

    def _handle_input(self, key):
        if self.ui.show_text_input:
            return self._handle_text_input(key)
        else:
            return self._handle_normal_input(key)

    def _handle_text_input(self, key):
        if key == 13 or key == 10:  # Enter
            movement_name = self.ui.input_text.upper()
            self.ui.input_text = ""
            self.ui.show_text_input = False
            if self.ui.input_action == "delete":
                # 🔥 CORREÇÃO: Usar o método correto de deleção
                result = self.core.delete_movement(movement_name)
                self.ui.set_error(result)
                self.ui.input_action = ""
            else:
                self.core.new_movement_name = movement_name
                self.ui.set_error(f"Sinal '{self.core.new_movement_name}' definido. Pressione 'R' para gravar.")
        elif key == 8:  # Backspace
            self.ui.input_text = self.ui.input_text[:-1]
        elif key == 27:  # ESC
            self.ui.show_text_input = False
            self.ui.input_text = ""
            self.ui.input_action = ""
        elif key != 255 and key != 0 and 32 <= key <= 126:
            self.ui.input_text += chr(key)
        return True
    
    def _handle_normal_input(self, key):
        if key == 27:  # ESC
            if self.core.mode == "treino":
                self.core.cancel_training()
                self.ui.set_error("Treino cancelado!")
                
        elif key == ord("m"):
            self._should_switch_to_gesture = True
            return False
            
        elif key == ord("c"):
            self.core.current_word = ""
            self.core.current_prediction = ""
            self.ui.set_error("Texto limpo!")
            
        elif key == ord("t"):
            self.core.start_training_mode()
            self.ui.set_error("Modo Treino ativado. Pressione 'N' para definir nome do sinal.")
            
        elif key == ord("n") and self.core.mode == "treino":
            self.ui.show_text_input = True
            self.ui.input_prompt = "Digite o nome do sinal de Libras:"
            self.ui.input_action = ""
            
        elif key == ord("s") and self.core.mode == "treino":
            result = self.core.save_movement()
            self.ui.set_error(result)
            
        elif key == ord("d") and self.core.mode == "teste":
            self.ui.show_text_input = True
            self.ui.input_prompt = "Digite o nome do sinal para deletar:"
            self.ui.input_action = "delete"
            
        elif key == ord("r"):
            if self.core.mode == "treino":
                if self.core.is_recording:
                    self.core.stop_recording()
                    self.ui.set_error("⏸️ Gravação pausada")
                else:
                    if self.core.new_movement_name:
                        if self.core.start_recording():
                            self.ui.set_error("🔴 GRAVANDO amostras...")
                        else:
                            self.ui.set_error("❌ Erro ao iniciar gravação")
                    else:
                        self.ui.set_error("❌ Defina um nome para o sinal primeiro (tecla N)")
            else:
                self.ui.set_error("❌ Entre no modo treino primeiro (tecla T)")
                
        elif key == ord("q"):
            return False
            
        elif key == ord("+") or key == ord("="):
            self._increase_resolution()
        elif key == ord("-") or key == ord("_"):
            self._decrease_resolution()
            
        return True

    def _increase_resolution(self):
        current_index = CONFIG["current_resolution_index"]
        if current_index > 0:
            new_index = current_index - 1
            new_width, new_height = CONFIG["resolution_options"][new_index]
            
            if self.core.change_resolution(new_width, new_height):
                CONFIG["current_resolution_index"] = new_index
                CONFIG["camera_resolution"] = (new_width, new_height)
                self.ui.set_error(f"📈 Resolução aumentada para {new_width}x{new_height}")
            else:
                self.ui.set_error("❌ Erro ao aumentar resolução")

    def _decrease_resolution(self):
        current_index = CONFIG["current_resolution_index"]
        if current_index < len(CONFIG["resolution_options"]) - 1:
            new_index = current_index + 1
            new_width, new_height = CONFIG["resolution_options"][new_index]
            
            if self.core.change_resolution(new_width, new_height):
                CONFIG["current_resolution_index"] = new_index
                CONFIG["camera_resolution"] = (new_width, new_height)
                self.ui.set_error(f"📉 Resolução diminuída para {new_width}x{new_height}")
            else:
                self.ui.set_error("❌ Erro ao diminuir resolução")

def main():
    db = DatabaseManager()
    current_screen = "gesture"

    while True:
        if current_screen == "gesture":
            app = GestureApp(db)
            result = app.run()
            if result == "movement":
                current_screen = "movement"
            else:
                break
        elif current_screen == "movement":
            app = MovementApp(db)
            result = app.run()
            if result == "gesture":
                current_screen = "gesture"
            else:
                break

    db.close()


if __name__ == "__main__":
    main()