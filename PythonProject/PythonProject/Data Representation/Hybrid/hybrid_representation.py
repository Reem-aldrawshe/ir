# hybrid_representation.py

import os
import joblib
import numpy as np
from tqdm import tqdm

def custom_tokenizer(text):
    """محلل بسيط لتقسيم النص حسب المسافات"""
    return text.split()

def build_hybrid_representation_in_chunks_joblib(
    tfidf_path,
    bert_path,
    save_path,
    chunk_size=5000
):
    """
    دمج تمثيلات TF-IDF وBERT وحفظها على دفعات باستخدام joblib
    """
    print("🚀 بناء التمثيل الهجين باستخدام TF-IDF وBERT بشكل دفعات (Chunks)...")

    # تحميل البيانات
    tfidf_data = joblib.load(tfidf_path)
    bert_data = joblib.load(bert_path)

    tfidf_matrix = tfidf_data["tfidf_matrix"]
    bert_matrix = np.array(bert_data["embeddings_matrix"], dtype=np.float32)
    doc_ids = tfidf_data["doc_ids"]

    if doc_ids != bert_data["doc_ids"]:
        raise ValueError("❌ doc_ids غير متطابقة بين TF-IDF وBERT!")

    total_docs = len(doc_ids)
    os.makedirs(save_path, exist_ok=True)

    for start in tqdm(range(0, total_docs, chunk_size), desc="🧩 Processing Chunks"):
        end = min(start + chunk_size, total_docs)

        tfidf_chunk = tfidf_matrix[start:end]
        bert_chunk = bert_matrix[start:end]
        chunk_doc_ids = doc_ids[start:end]

        chunk_data = {
            "tfidf_chunk": tfidf_chunk,
            "bert_chunk": bert_chunk,
            "doc_ids": chunk_doc_ids
        }

        chunk_file = os.path.join(save_path, f"hybrid_chunk_{start}_{end}.joblib")
        joblib.dump(chunk_data, chunk_file, compress=3)

    print(f"✅ تم حفظ جميع التمثيلات الهجينة بصيغة joblib في: {save_path}")

# نقطة الدخول لتشغيل السكريبت مباشرة
if __name__ == "__main__":
    # مثال على التنفيذ - عدّل المسارات حسب الحاجة
    tfidf_path = "antique/train/doc/tfidf.joblib"
    bert_path = "antique/train/doc/bert_embedding.joblib"
    save_path = "antique/train/doc/hybrid_chunks"

    build_hybrid_representation_in_chunks_joblib(tfidf_path, bert_path, save_path)
