
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

BERT_PATHS = {
    "antique": r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\Bert\antique\train\doc\bert_embedding.joblib",
    "beir": r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\Bert\beir\quora\test\doc\bert_embedding.joblib"
}

MONGO_COLLECTIONS = {
    "antique": "documents_test",
    "beir": "documents_quora_test"
}

client = MongoClient("mongodb://localhost:27017/")
db = client["ir_project"]

from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def bert_search(query: str, dataset: str, top_k: int = 10):

    bert_data = joblib.load(BERT_PATHS[dataset])

    doc_embeddings = np.vstack(bert_data["embeddings_matrix"])
    doc_ids = bert_data["doc_ids"]

    query_emb = bert_model.encode(query, convert_to_numpy=True).reshape(1, -1)

    scores = cosine_similarity(query_emb, doc_embeddings).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    coll = db[MONGO_COLLECTIONS[dataset]]
    id_to_doc = {
        doc["doc_id"]: doc
        for doc in coll.find(
            {"doc_id": {"$in": [doc_ids[idx] for idx in top_indices]}},
            {"_id": 0, "doc_id": 1, "original_text": 1, "clean_text": 1}
        )
    }

    results = []
    for idx in top_indices:
        doc_id = doc_ids[idx]
        mongo_doc = id_to_doc.get(doc_id, {})
        results.append({
            "doc_id": doc_id,
            "original_text": mongo_doc.get("original_text", "[Not Found]"),
            "clean_text": mongo_doc.get("clean_text", "[Not Found]"),
            "score": float(round(scores[idx], 4))
        })

    return results
