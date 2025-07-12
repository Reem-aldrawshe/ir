
import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from glob import glob
from sentence_transformers import SentenceTransformer

HYBRID_PATHS = {
    "antique": r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\Hybrid\antique\train\chunks",
    "beir": r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\Hybrid\beir\quora\test\chunks"
}

MONGO_COLLECTIONS = {
    "antique": "documents_test",
    "beir": "documents_quora_test"
}

client = MongoClient("mongodb://localhost:27017/")
db = client["ir_project"]


bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def hybrid_search(query: str, dataset: str, top_k: int = 10, alpha=0.5):
    """
    بحث هجين: يجمع بين تشابه TF-IDF و BERT.
    """
    chunks_dir = HYBRID_PATHS[dataset]
    chunk_files = sorted(glob(os.path.join(chunks_dir, "*.joblib")))

    
    query_bert_vec = bert_model.encode(query, convert_to_numpy=True).reshape(1, -1)

    all_scores = []
    all_doc_ids = []

    for chunk_file in chunk_files:
        chunk_data = joblib.load(chunk_file)

        tfidf_chunk = chunk_data["tfidf_chunk"]
        bert_chunk = np.array(chunk_data["bert_chunk"], dtype=np.float32)
        doc_ids_chunk = chunk_data["doc_ids"]

        
        tfidf_query_vec = np.zeros((1, tfidf_chunk.shape[1]))  

        tfidf_scores = cosine_similarity(tfidf_query_vec, tfidf_chunk).flatten()

        bert_scores = cosine_similarity(query_bert_vec, bert_chunk).flatten()

        hybrid_scores = alpha * tfidf_scores + (1 - alpha) * bert_scores

        all_scores.extend(hybrid_scores)
        all_doc_ids.extend(doc_ids_chunk)

    all_scores = np.array(all_scores)
    top_indices = np.argsort(all_scores)[::-1][:top_k]
    top_doc_ids = [all_doc_ids[idx] for idx in top_indices]

    coll = db[MONGO_COLLECTIONS[dataset]]
    id_to_doc = {
        doc["doc_id"]: doc
        for doc in coll.find(
            {"doc_id": {"$in": top_doc_ids}},
            {"_id": 0, "doc_id": 1, "original_text": 1, "clean_text": 1}
        )
    }

    results = []
    for idx in top_indices:
        doc_id = all_doc_ids[idx]
        mongo_doc = id_to_doc.get(doc_id, {})
        results.append({
            "doc_id": doc_id,
            "original_text": mongo_doc.get("original_text", "[Not Found]"),
            "clean_text": mongo_doc.get("clean_text", "[Not Found]"),
            "score": float(round(all_scores[idx], 4))
        })

    return results
