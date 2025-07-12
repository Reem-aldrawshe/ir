import joblib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def search_faiss_vector_store(query_text, faiss_index_path, doc_ids_path, top_k=10):
    index = faiss.read_index(faiss_index_path)
    doc_ids = joblib.load(doc_ids_path)

    bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = bert_model.encode(query_text, convert_to_numpy=True).astype('float32')
    query_embedding = query_embedding.reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(len(indices[0])):
        doc_idx = indices[0][i]
        results.append((doc_ids[doc_idx], float(distances[0][i])))

    return results

if __name__ == "__main__":
    results = search_faiss_vector_store(
        "how to learn deep learning models",
        r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\VectorStore\beir\quora\test\faiss_index_beir.faiss",
        r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\VectorStore\beir\quora\test\faiss_index_beir_doc_ids.joblib",
        top_k=5
    )
    print("🔍 Results:")
    for i, (doc_id, dist) in enumerate(results, 1):
        print(f"{i}. Doc ID: {doc_id} | Distance: {dist:.4f}")
