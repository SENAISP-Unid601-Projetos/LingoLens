# app_gradio.py — VERSÃO FINAL OFICIAL (FUNCIONA NO NAVEGADOR E NO CELULAR)

import gradio as gr
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.core.GestureApp import GestureApp
from src.utils.Extract_landmarks import extract_landmarks

print("Carregando LingoLens...")
app = GestureApp()
app.mode = "teste"
app.min_time_between_any_letter = 0.3   # bom equilíbrio
print("LingoLens Web pronto!")

def process_frame(frame):
    if frame is None:
        return frame, app.current_word

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = app.hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        app.mp_drawing.draw_landmarks(frame, hand, app.mp_hands.HAND_CONNECTIONS)

        landmarks = extract_landmarks(hand, frame.shape)
        if landmarks:
            lm = np.array(landmarks, dtype=np.float32)
            if app.smooth_landmarks is None:
                app.smooth_landmarks = lm
            else:
                app.smooth_landmarks = 0.7 * lm + 0.3 * app.smooth_landmarks

            app.sequence_buffer.append(app.smooth_landmarks.tolist())
            if len(app.sequence_buffer) > 38:
                app.sequence_buffer = app.sequence_buffer[-38:]

            app._handle_motion_state(app.smooth_landmarks.tolist(), frame)

    app._draw_ui(frame)  # agora sem cv2.imshow()

    current_text = getattr(app, 'current_word', '')
    return frame, current_text.upper()

with gr.Blocks() as demo:
    gr.Markdown("# ✋ LingoLens – Tradutor de Libras em Tempo Real (Web 2025)")
    with gr.Row():
        video = gr.Image(sources=["webcam"], streaming=True, label="Câmera", height=480)
        output = gr.Image(label="Reconhecimento")
    texto = gr.Textbox(label="Palavra formada", value="", interactive=False, scale=2)

    video.change(process_frame, video, [output, texto])

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    inbrowser=True,
    share=False
)