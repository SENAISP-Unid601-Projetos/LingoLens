from GestureApp import GestureApp
from MovementApp import MovementApp
from GestureTrainer import GestureTrainer  # Adicione esta linha
from Database_manager import DatabaseManager


def main():
    # Inicializa banco de dados
    db = DatabaseManager()
    current_screen = "gesture"

    while True:
        if current_screen == "gesture":
            # Inicializa app de gestos
            gesture_app = GestureApp(db)
            result = gesture_app.run()  # retorna "movement" se apertar m

            if result == "movement":
                current_screen = "movement"
            else:
                break  # sair do programa

        elif current_screen == "movement":
            # Inicializa app de movimentos
            movement_app = MovementApp(db)
            result = movement_app.run()  # retorna "gesture" se apertar m

            if result == "gesture":
                current_screen = "gesture"
            else:
                break  # sair do programa

    # Fecha banco de dados ao final
    db.close()


if __name__ == "__main__":
    main()