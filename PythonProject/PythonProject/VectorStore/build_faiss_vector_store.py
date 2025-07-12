import joblib
import numpy as np
import faiss
import os

def build_and_save_faiss_index(bert_embeddings_path, output_index_path):
    bert_data = joblib.load(bert_embeddings_path)
    doc_embeddings = np.array(bert_data["embeddings_matrix"], dtype='float32')
    doc_ids = bert_data["doc_ids"]

    if doc_embeddings.ndim != 2:
        raise ValueError(f"Dim mismatch: {doc_embeddings.shape}. Expected (num_docs, embedding_dim).")

    dimension = doc_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    faiss.write_index(index, output_index_path)
    joblib.dump(doc_ids, output_index_path.replace(".faiss", "_doc_ids.joblib"))
    print(f"✅ Saved FAISS index: {output_index_path}")
    print(f"✅ Saved doc_ids: {output_index_path.replace('.faiss', '_doc_ids.joblib')}")

if __name__ == "__main__":
    build_and_save_faiss_index(
        r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\Bert\antique\train\doc\bert_embedding.joblib",
        r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\VectorStore\antique\train\faiss_index_antique.faiss"
    )

    build_and_save_faiss_index(
        r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\Bert\beir\quora\test\doc\bert_embedding.joblib",
        r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\VectorStore\beir\quora\test\faiss_index_beir.faiss"
    )
