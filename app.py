import streamlit as st
import json
import torch
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Chatbot Fakultas Teknik UNSADA", page_icon="🤖")

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def load_faq_data():
    with open('train_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

model = load_model()
faq_data = load_faq_data()

# Siapkan pertanyaan dan jawaban
all_questions = []
all_answers = []

for item in faq_data:
    question = item['question']
    answer = item['answer']
    all_questions.append(question)
    all_answers.append(answer)

    for variation in item.get('variations', []):
        all_questions.append(variation)
        all_answers.append(answer)

# Encode semua pertanyaan sekali saja
question_embeddings = model.encode(all_questions, convert_to_tensor=True, normalize_embeddings=True)

def get_response(user_input):
    input_embedding = model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)
    similarity_scores = util.cos_sim(input_embedding, question_embeddings)[0]
    best_idx = torch.argmax(similarity_scores).item()
    best_score = similarity_scores[best_idx].item()

    threshold = 0.60
    if best_score >= threshold:
        return all_answers[best_idx]
    else:
        return "Maaf, saya tidak memahami pertanyaan Anda. Silakan ulangi dengan pertanyaan yang lebih spesifik."

# UI Chatbot
st.title("🤖 Chatbot Fakultas Teknik UNSADA")
st.markdown("Silakan ketik pertanyaan Anda di bawah ini.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Tanyakan sesuatu...")
if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# Tampilkan riwayat chat
for role, message in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.write(message)
