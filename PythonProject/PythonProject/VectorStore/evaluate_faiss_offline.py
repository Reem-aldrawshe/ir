import os
import joblib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ir_datasets
from collections import defaultdict
from tqdm import tqdm

FAISS_PATHS = {
    "antique": {
        "index": r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\VectorStore\antique\train\faiss_index_antique.faiss",
        "doc_ids": r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\VectorStore\antique\train\faiss_index_antique_doc_ids.joblib"
    },
    "beir": {
        "index": r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\VectorStore\beir\quora\test\faiss_index_beir.faiss",
        "doc_ids": r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\VectorStore\beir\quora\test\faiss_index_beir_doc_ids.joblib"
    }
}

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if not retrieved_k: return 0.0
    return len([doc for doc in retrieved_k if doc in relevant]) / k

def recall_at_k(retrieved, relevant, k):
    if not relevant: return 0.0
    return len([doc for doc in retrieved if doc in relevant]) / len(relevant)

def average_precision(retrieved, relevant, k):
    score, hits = 0.0, 0
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0

def dcg(retrieved, relevant, k):
    return sum([(1 if retrieved[i] in relevant else 0) / np.log2(i+2) for i in range(min(len(retrieved), k))])

def idcg(relevant, k):
    return sum([1 / np.log2(i+2) for i in range(min(len(relevant), k))])

def ndcg_at_k(retrieved, relevant, k):
    dcg_val = dcg(retrieved, relevant, k)
    idcg_val = idcg(relevant, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def evaluate_faiss(dataset_name: str, top_k=10):
    paths = FAISS_PATHS[dataset_name]
    index = faiss.read_index(paths["index"])
    doc_ids = joblib.load(paths["doc_ids"])

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if dataset_name == "antique":
        dataset = ir_datasets.load("antique/test/non-offensive")
    elif dataset_name == "beir":
        dataset = ir_datasets.load("beir/quora/test")
    else:
        raise ValueError("Unsupported dataset")

    qrels = defaultdict(set)
    for qrel in dataset.qrels_iter():
        if int(qrel.relevance) > 0:
            qrels[qrel.query_id].add(qrel.doc_id)

    precisions, recalls, maps, ndcgs = [], [], [], []

    for query in tqdm(dataset.queries_iter(), desc=f"Evaluating FAISS on {dataset_name}"):
        query_text = query.text
        query_id = query.query_id
        relevant = qrels[query_id]

        query_vec = model.encode(query_text, convert_to_numpy=True).astype('float32').reshape(1, -1)
        distances, indices = index.search(query_vec, top_k)
        retrieved = [doc_ids[i] for i in indices[0]]

        precisions.append(precision_at_k(retrieved, relevant, top_k))
        recalls.append(recall_at_k(retrieved, relevant, top_k))
        maps.append(average_precision(retrieved, relevant, top_k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, top_k))

    evaluation_results = {
        "Precision@10": round(np.mean(precisions), 4),
        "Recall@10": round(np.mean(recalls), 4),
        "MAP@10": round(np.mean(maps), 4),
        "NDCG@10": round(np.mean(ndcgs), 4),
    }

    return evaluation_results


if __name__ == "__main__":
    dataset_name = input("📊 Enter dataset (antique / beir): ").strip()
    results = evaluate_faiss(dataset_name)
    print("\n📈 FAISS Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v}")
