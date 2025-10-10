import argparse
from Gesture_recognizer import GestureRecognizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar e testar gestos de Libras")
    parser.add_argument("--specific_letter", type=str, default=None, help="Nome do gesto (ex.: A, MAE)")
    parser.add_argument("--gesture_type", type=str, default="letter", choices=["letter", "word", "movement"], help="Tipo de gesto")
    args = parser.parse_args()
    app = GestureRecognizer(specific_letter=args.specific_letter, gesture_type=args.gesture_type)
    app.run()