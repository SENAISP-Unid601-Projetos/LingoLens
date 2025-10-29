from GestureApp import GestureApp
import sys
import os

# Adiciona o diretório do app ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        app = GestureApp(gesture_type="letter")
        app.run()
    except Exception as e:
        print(f"[FATAL] Erro crítico: {e}")
        import traceback
        traceback.print_exc()
        input("Pressione Enter para sair...")