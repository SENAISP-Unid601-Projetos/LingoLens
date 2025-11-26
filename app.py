# app.py — coloca na raiz do projeto (mesmo nível do src/, data/, etc.)

import streamlit as st
from src.core.GestureApp import GestureApp
import cv2

# === Configuração da página ===
st.set_page_config(
    page_title="LingoLens – Tradutor de Datilologia Libras 2025",
    page_icon="✋",
    layout="centered"
)

# === Título e descrição ===
st.title("✋ LingoLens – Tradutor de Libras em Tempo Real")
st.markdown("""
**Abra no celular • Ligue a câmera • Faça os gestos**  
Reconhecimento de datilologia estática e dinâmica (G, J, Z, etc.)  
Projeto de TCC 2025 – Feito com ❤️ para inclusão
""")

# === Inicializa o app só uma vez ===
if 'lingolens' not in st.session_state:
    with st.spinner("Inicializando LingoLens... (primeira vez demora um pouco)"):
        st.session_state.lingolens = GestureApp()
        st.session_state.placeholder = st.empty()

app = st.session_state.lingolens
placeholder = st.session_state.placeholder

# === Loop principal (roda a cada frame) ===
ret, frame = app.cap.read()
if not ret:
    st.error("Câmera não encontrada. Verifique a permissão.")
    st.stop()

frame = cv2.flip(frame, 1)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = app.hands.process(rgb)

if results.multi_hand_landmarks:
    hand = results.multi_hand_landmarks[0]
    app.mp_drawing.draw_landmarks(frame, hand, app.mp_hands.HAND_CONNECTIONS)
    landmarks = app.extract_landmarks(hand, frame.shape)
    if landmarks:
        # Suavização
        lm = app.smooth_landmarks = 0.65 * np.array(landmarks) + 0.35 * (app.smooth_landmarks or np.array(landmarks))
        app.sequence_buffer.append(lm.tolist())
        
        if app.mode == "treino" and app.new_gesture_name:
            app._capture_training_sample(lm.tolist(), frame)
        else:
            app._handle_motion_state(lm.tolist(), frame)

# Desenha a interface linda que você criou
app._draw_ui(frame)

# Mostra o frame no Streamlit
placeholder.image(frame, channels="BGR", use_column_width=True)

# Rodapé
st.caption("LingoLens 2025 • Feito por [Seu Nome] • Acesso em tempo real via navegador")