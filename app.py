import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch

# ===========================
# Konfigurasi dasar
# ===========================
st.set_page_config(page_title="Fakultas Teknik UNSADA", layout="wide")

# ===========================
# State untuk navigasi
# ===========================
if "page" not in st.session_state:
    st.session_state.page = "landing"

# ===========================
# Fungsi Chatbot
# ===========================
def chatbot_ui():
    st.markdown("""
        <style>
            .chat-bubble {
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                max-width: 80%;
                display: inline-block;
            }
            .user {
                background-color: #1e3a8a;
                color: white;
                align-self: flex-end;
            }
            .bot {
                background-color: #e5e7eb;
                color: black;
                align-self: flex-start;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🤖 Chatbot Fakultas Teknik UNSADA")
    st.markdown("Ketik pertanyaan Anda di bawah ini:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role, text = msg
        css_class = "user" if role == "user" else "bot"
        st.markdown(f'<div class="chat-bubble {css_class}">{text}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Tulis pertanyaan Anda...")
    if user_input:
        st.session_state.messages.append(("user", user_input))

        # Proses dengan model
        with open("train_data.json", "r", encoding="utf-8") as f:
            faq_data = json.load(f)

        all_questions = []
        all_answers = []
        for item in faq_data:
            all_questions.append(item['question'])
            all_answers.append(item['answer'])
            for var in item.get("variations", []):
                all_questions.append(var)
                all_answers.append(item['answer'])

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        question_embeddings = model.encode(all_questions, convert_to_tensor=True, normalize_embeddings=True)
        input_embedding = model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)
        similarity_scores = util.cos_sim(input_embedding, question_embeddings)[0]
        best_idx = torch.argmax(similarity_scores).item()
        best_score = similarity_scores[best_idx].item()

        if best_score >= 0.60:
            response = all_answers[best_idx]
        else:
            response = "Maaf, saya tidak memahami pertanyaan Anda. Silakan ulangi dengan kata lain."

        st.session_state.messages.append(("bot", response))

    st.button("🔙 Kembali ke Beranda", on_click=lambda: st.session_state.update({"page": "landing", "messages": []}))

# ===========================
# Fungsi Landing Page
# ===========================
def landing_page():
    st.image("https://unsada.ac.id/sites/default/files/2020-11/logo-unsada.png", width=100)
    st.markdown("""
        # Selamat Datang di Fakultas Teknik UNSADA
        Fakultas Teknik Universitas Darma Persada memiliki 5 program studi unggulan dan komitmen terhadap pendidikan teknik berkualitas.
        
        ### Program Studi:
        - Teknologi Informasi
        - Sistem Informasi
        - Teknik Mesin
        - Teknik Industri
        - Teknik Elektro
    """)

    st.markdown("---")
    st.markdown("### Tentang Fakultas")
    st.write("Fakultas Teknik Universitas Darma Persada berkomitmen untuk menyediakan pendidikan teknik berkualitas tinggi yang relevan dengan kebutuhan industri.")

    st.markdown("---")
    st.markdown("### Hubungi Kami")
    st.write("📧 teknik@unsada.ac.id  |  📞 (021) 8650-896")

    st.markdown("---")
    st.button("💬 Mulai Chatbot", on_click=lambda: st.session_state.update({"page": "chatbot"}))

# ===========================
# Navigasi Utama
# ===========================
if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "chatbot":
    chatbot_ui()
