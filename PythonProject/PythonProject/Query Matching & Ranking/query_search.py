import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

def load_models():
    # تحميل موديل BERT
    bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # تحميل بيانات TF-IDF من المسار الصحيح
    tfidf_path = r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\TF-IDF\antique\train\doc\tfidf_data.joblib"
    tfidf_data = joblib.load(tfidf_path)
    tfidf_vectorizer = tfidf_data["vectorizer"]
    doc_tfidf_matrix = tfidf_data["tfidf_matrix"]
    tfidf_doc_ids = tfidf_data["doc_ids"]

    # تحميل بيانات BERT من المسار الصحيح
    bert_path = r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\Bert\antique\train\doc\bert_embedding.joblib"
    bert_data = joblib.load(bert_path)
    doc_bert_embeddings = np.vstack(bert_data["embeddings_matrix"])
    bert_doc_ids = bert_data.get("doc_ids", tfidf_doc_ids)

    return bert_model, tfidf_vectorizer, doc_tfidf_matrix, tfidf_doc_ids, doc_bert_embeddings, bert_doc_ids

def compute_similarity(query_embedding, doc_embeddings, top_k=10):
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    sorted_indices = top_indices[np.argsort(-similarities[top_indices])]
    return [(i, similarities[i]) for i in sorted_indices]

def save_results(query, method, results, doc_ids, output_path):
    formatted = {
        "query": query,
        "method": method,
        "results": [
            {"doc_id": doc_ids[idx], "score": float(round(score, 6))} for idx, score in results
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    print(f"✅ تم حفظ النتائج في: {output_path}")

def run_query_search(user_query, method="tfidf", top_k=10):
    bert_model, tfidf_vectorizer, doc_tfidf_matrix, tfidf_doc_ids, doc_bert_embeddings, bert_doc_ids = load_models()

    # توحيد doc_ids بين TF-IDF وBERT
    common_doc_ids = list(set(tfidf_doc_ids).intersection(set(bert_doc_ids)))
    common_doc_ids.sort()

    tfidf_idx_map = {doc_id: idx for idx, doc_id in enumerate(tfidf_doc_ids)}
    bert_idx_map = {doc_id: idx for idx, doc_id in enumerate(bert_doc_ids)}

    tfidf_indices = [tfidf_idx_map[doc_id] for doc_id in common_doc_ids]
    bert_indices = [bert_idx_map[doc_id] for doc_id in common_doc_ids]

    doc_tfidf_matrix_common = doc_tfidf_matrix[tfidf_indices]
    doc_bert_embeddings_common = doc_bert_embeddings[bert_indices]

    method = method.lower().strip()
    if method == "tfidf":
        query_vec = tfidf_vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vec, doc_tfidf_matrix_common)[0]
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        sorted_indices = top_indices[np.argsort(-similarities[top_indices])]
        results = [(i, similarities[i]) for i in sorted_indices]
    elif method == "bert":
        query_vec = bert_model.encode(user_query)
        results = compute_similarity(query_vec, doc_bert_embeddings_common, top_k)
    elif method == "hybrid":
        alpha = 0.5
        tfidf_query = tfidf_vectorizer.transform([user_query])
        bert_query = bert_model.encode(user_query)
        tfidf_sim = cosine_similarity(tfidf_query, doc_tfidf_matrix_common)[0]
        bert_sim = cosine_similarity([bert_query], doc_bert_embeddings_common)[0]
        hybrid_sim = alpha * tfidf_sim + (1 - alpha) * bert_sim
        top_indices = np.argpartition(hybrid_sim, -top_k)[-top_k:]
        sorted_indices = top_indices[np.argsort(-hybrid_sim[top_indices])]
        results = [(i, hybrid_sim[i]) for i in sorted_indices]
    else:
        raise ValueError("❌ نوع التمثيل غير مدعوم. استخدم: tfidf / bert / hybrid")

    save_results(user_query, method, results, common_doc_ids, "user_query_result.json")
    return results
