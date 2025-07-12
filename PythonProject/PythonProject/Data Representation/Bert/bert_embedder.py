# bert_embedder.py

import os
import joblib
import torch
from tqdm import tqdm
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel

# تحميل النموذج
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# استخدام GPU إذا كان متاحاً
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_bert_embedding(text):
    """تحويل نص إلى تمثيل BERT باستخدام CLS token"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    return embeddings.squeeze().cpu().numpy()

def process_bert_from_mongodb(dataset_name, collection_name):
    """استخراج التمثيلات من مجموعة MongoDB"""
    print(f"🚀 Processing BERT embeddings from MongoDB collection: {collection_name}...")

    # الاتصال بقاعدة البيانات
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ir_project"]
    collection = db[collection_name]

    # جلب الوثائق
    documents = list(collection.find({}, {"_id": 0, "doc_id": 1, "original_text": 1}))
    documents = [doc for doc in documents if doc.get("original_text", "").strip()]

    if not documents:
        print("❌ لا توجد نصوص أصلية متاحة للمعالجة.")
        return

    doc_ids = [doc["doc_id"] for doc in documents]
    texts = [doc["original_text"] for doc in documents]

    # الحصول على التمثيلات
    all_embeddings = []
    for text in tqdm(texts, desc="Embedding"):
        emb = get_bert_embedding(text)
        all_embeddings.append(emb)

    # حفظ النتائج
    embedding_data = {
        "doc_ids": doc_ids,
        "embeddings_matrix": all_embeddings,
        "model_name": MODEL_NAME
    }

    save_path = os.path.join(dataset_name.replace("/", os.sep), "doc")
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "bert_embedding.joblib")
    joblib.dump(embedding_data, save_file)

    print(f"✅ BERT embeddings saved to: {save_file}")

# نقطة الدخول الرئيسية
if __name__ == "__main__":
    # أمثلة تنفيذ عند التشغيل المباشر
    process_bert_from_mongodb("beir/quora/test", "documents_quora_test")
