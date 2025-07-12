
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
from pymongo import MongoClient

TFIDF_PATHS = {
    "antique": r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\TF-IDF\antique\train\doc\tfidf_data.joblib",
    "beir": r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\TF-IDF\beir\quora\test\doc\tfidf_data.joblib"
}

MONGO_COLLECTIONS = {
    "antique": "documents_test",
    "beir": "documents_quora_test"
}

client = MongoClient("mongodb://localhost:27017/")
db = client["ir_project"]


def tfidf_search(query: str, dataset: str, top_k: int = 10):

    tfidf_data = joblib.load(TFIDF_PATHS[dataset])

    vectorizer = tfidf_data["vectorizer"]
    tfidf_matrix = tfidf_data["tfidf_matrix"]
    doc_ids = tfidf_data["doc_ids"]

    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
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
