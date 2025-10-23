from GestureApp import GestureApp
from MovementApp import MovementApp
from Database_manager import DatabaseManager


def main():
    db = DatabaseManager()
    current_screen = "gesture"

    while True:
        if current_screen == "gesture":
            gesture_app = GestureApp(db)
            result = gesture_app.run()

            if result == "movement":
                current_screen = "movement"
            else:
                break

        elif current_screen == "movement":
            movement_app = MovementApp(db)
            result = movement_app.run()

            if result == "gesture":
                current_screen = "gesture"
            else:
                break

    db.close()


if __name__ == "__main__":
    main()