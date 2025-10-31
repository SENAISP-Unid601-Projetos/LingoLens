import cv2
import numpy as np
from core.LibrasCore import LibrasCore
from ui.LibrasUI import LibrasUI
from Database_manager import DatabaseManager
from Config import CONFIG


class LibrasApp:
    def __init__(self, db):
        self.core = LibrasCore(db)
        self.ui = LibrasUI()
        self._should_exit = False

    def run(self):
        print("=" * 60)
        print("        TRADUTOR DE LIBRAS - SISTEMA UNIFICADO")
        print("=" * 60)
        print("Modos disponíveis:")
        print("  - Letras: Reconhecimento de A-Z (soletração)")
        print("  - Palavras: Sinais completos de palavras")
        print("  - Frases: Sinais complexos com movimento")
        print("=" * 60)
        print("[INFO] Teclas: Q=Sair | C=Limpar | M=Mudar Modo | T=Treino | S=Salvar | N=Nome | +/-:Resolução")

        window_width, window_height = CONFIG["window_size"]
        
        while not self._should_exit:
            try:
                ret, frame = self.core.cap.read()
                if not ret:
                    self.ui.set_error("❌ Falha ao capturar frame da câmera!")
                    break

                # Processa frame (lógica de Libras)
                image, landmarks_list = self.core.process_frame(frame)
                
                # Lógica de treino
                if self.core.mode == "treino" and landmarks_list:
                    self.core.add_training_sample(landmarks_list[0])

                # Desenha UI
                image = self.ui.draw_ui(image, self.core)
                display_image = self.ui.resize_for_display(image)
                
                # 🔥 JANELA REDIMENSIONÁVEL
                cv2.namedWindow("Tradutor de Libras", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Tradutor de Libras", window_width, window_height)
                cv2.moveWindow("Tradutor de Libras", 100, 50)
                
                cv2.imshow("Tradutor de Libras", display_image)

                # Processa input
                key = cv2.waitKey(1) & 0xFF
                should_continue = self._handle_input(key)
                if not should_continue:
                    break

                # Atualiza UI
                self.ui.update_error_timer()
                
            except Exception as e:
                print(f"❌ ERRO: {e}")
                import traceback
                traceback.print_exc()
                self.ui.set_error(f"Erro: {str(e)}")

        self.core.cleanup()
        cv2.destroyAllWindows()
        return None

    def _handle_input(self, key):
        if self.ui.show_text_input:
            return self._handle_text_input(key)
        else:
            return self._handle_normal_input(key)

    def _handle_text_input(self, key):
        if key == 13 or key == 10:  # Enter
            text = self.ui.input_text.upper().strip()
            self.ui.input_text = ""
            self.ui.show_text_input = False
            
            if self.ui.input_action == "train_name":
                if text:
                    self.core.new_gesture_name = text
                    self.ui.set_error(f"🎯 Treinando: '{text}' - Capture amostras pressionando 'S'")
                else:
                    self.ui.set_error("❌ Nome não pode estar vazio!")
                    
            elif self.ui.input_action == "train_type":
                if text in ["LETRA", "PALAVRA", "FRASE"]:
                    gesture_type = "letter" if text == "LETRA" else "word" if text == "PALAVRA" else "sentence"
                    self.core.start_training_mode(gesture_type)
                    self.ui.set_error(f"🔧 Modo treino: {text}")
                    # Pedir nome depois de escolher o tipo
                    self.ui.show_text_input = True
                    self.ui.input_prompt = f"Nome da {text.lower()}:"
                    self.ui.input_action = "train_name"
                else:
                    self.ui.set_error("❌ Tipo inválido! Use: LETRA, PALAVRA ou FRASE")
            
            elif self.ui.input_action == "delete":
                if text:
                    result = self.core.delete_gesture(text)
                    self.ui.set_error(result)
                else:
                    self.ui.set_error("❌ Nome não pode estar vazio!")
                self.ui.input_action = ""
                
        elif key == 8:  # Backspace
            self.ui.input_text = self.ui.input_text[:-1]
        elif key == 27:  # ESC
            self.ui.show_text_input = False
            self.ui.input_text = ""
            self.ui.input_action = ""
            if self.core.mode == "treino":
                self.core.cancel_training()
                self.ui.set_error("❌ Treino cancelado!")
        elif key != 255 and key != 0 and 32 <= key <= 126:
            self.ui.input_text += chr(key)
        
        return True

    def _handle_normal_input(self, key):
        if key == 27:  # ESC
            if self.core.mode == "treino":
                self.core.cancel_training()
                self.ui.set_error("❌ Modo treino cancelado.")
            else:
                self.ui.set_error("ℹ️  Pressione Q para sair")

        elif key == ord("q"):
            return False
            
        elif key == ord("m"):  # 🔥 MUDAR MODO DE RECONHECIMENTO
            current_index = CONFIG["recognition_modes"].index(self.core.recognition_mode)
            new_index = (current_index + 1) % len(CONFIG["recognition_modes"])
            new_mode = CONFIG["recognition_modes"][new_index]
            
            if self.core.switch_recognition_mode(new_mode):
                self.ui.set_error(f"🔁 Modo alterado para: {new_mode.upper()}")
            
        elif key == ord("c"):  # LIMPAR TEXTO
            self.core.clear_text()
            self.ui.set_error("🗑️  Texto limpo!")
            
        elif key == ord("t"):  # INICIAR TREINO
            if self.core.mode == "teste":
                self.ui.show_text_input = True
                self.ui.input_prompt = "Tipo de gesto (LETRA/PALAVRA/FRASE):"
                self.ui.input_action = "train_type"
            else:
                self.ui.set_error("❌ Finalize o treino atual primeiro!")
            
        elif key == ord("n") and self.core.mode == "treino":  # DEFINIR NOME
            self.ui.show_text_input = True
            self.ui.input_prompt = "Nome do gesto:"
            self.ui.input_action = "train_name"
            
        elif key == ord("s") and self.core.mode == "treino":  # SALVAR GESTO
            if self.core.new_gesture_name:
                result = self.core.save_gesture()
                self.ui.set_error(result)
            else:
                self.ui.set_error("❌ Defina um nome primeiro (tecla N)")
                
        elif key == ord("d") and self.core.mode == "teste":  # DELETAR GESTO
            self.ui.show_text_input = True
            self.ui.input_prompt = "Nome do gesto para deletar:"
            self.ui.input_action = "delete"
            
        elif key == ord("r"):  # RESETAR DETECÇÃO
            self.core.reset_detection()
            self.ui.set_error("🔄 Detecção resetada!")
            
        # 🔥 CONTROLES DE RESOLUÇÃO
        elif key == ord("+") or key == ord("="):
            self._increase_resolution()
        elif key == ord("-") or key == ord("_"):
            self._decrease_resolution()

        return True

    def _increase_resolution(self):
        """Aumenta a resolução da câmera"""
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
        """Diminui a resolução da câmera"""
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
    """Função principal do Tradutor de Libras"""
    db = DatabaseManager()
    
    try:
        # Criar e executar o aplicativo unificado de Libras
        app = LibrasApp(db)
        app.run()
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Garantir que os recursos sejam liberados
        db.close()
        cv2.destroyAllWindows()
        print("👋 Tradutor de Libras finalizado.")


if __name__ == "__main__":
    main()