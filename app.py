from flask import Flask, request, jsonify
import json
import torch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# ✅ Load model hasil fine-tuning
model = SentenceTransformer('model_finetuned_unsada')

# ✅ Load dataset FAQ
with open('train_data.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

# ✅ Siapkan daftar pertanyaan dan jawaban
all_questions = []
all_answers = []

for item in faq_data:
    question = item['question']
    answer = item['answer']

    # Tambahkan pertanyaan utama
    all_questions.append(question)
    all_answers.append(answer)

    # Tambahkan variasi (jika ada)
    for variation in item.get('variations', []):
        all_questions.append(variation)
        all_answers.append(answer)

# ✅ Encode semua pertanyaan (sekali saja)
question_embeddings = model.encode(all_questions, convert_to_tensor=True, normalize_embeddings=True)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Pesan kosong tidak dapat diproses."})

    # ✅ Encode input pengguna
    input_embedding = model.encode(user_input, convert_to_tensor=True, normalize_embeddings=True)

    # ✅ Hitung similarity
    similarity_scores = util.cos_sim(input_embedding, question_embeddings)[0]
    best_idx = torch.argmax(similarity_scores).item()
    best_score = similarity_scores[best_idx].item()

    # ✅ Threshold relevansi jawaban
    threshold = 0.60
    if best_score >= threshold:
        answer = all_answers[best_idx]
    else:
        answer = "Maaf, saya tidak memahami pertanyaan Anda. Silakan ulangi dengan pertanyaan yang lebih spesifik."

    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
