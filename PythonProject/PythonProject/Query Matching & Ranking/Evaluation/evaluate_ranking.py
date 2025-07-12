import json
import ir_datasets
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def evaluate_ranking(results_path, k=10):
    # تحميل qrels من BEIR Quora
    dataset = ir_datasets.load("beir/quora/test")
    qrels = defaultdict(set)
    for qrel in dataset.qrels_iter():
        if int(qrel.relevance) > 0:
            qrels[qrel.query_id].add(qrel.doc_id)

    # تحميل نتائج المطابقة من ملف JSON
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # دوال التقييم
    def precision_at_k(retrieved, relevant, k):
        retrieved_k = retrieved[:k]
        if not retrieved_k:
            return 0.0
        return len([doc for doc in retrieved_k if doc in relevant]) / k

    def recall_at_k(retrieved, relevant, k):
        retrieved_k = retrieved[:k]
        if not relevant:
            return 0.0
        return len([doc for doc in retrieved_k if doc in relevant]) / len(relevant)

    def average_precision(retrieved, relevant, k):
        score = 0.0
        hits = 0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in relevant:
                hits += 1
                score += hits / i
        return score / min(len(relevant), k) if relevant else 0.0

    def dcg(retrieved, relevant, k):
        return sum([(1 if retrieved[i] in relevant else 0) / np.log2(i + 2) for i in range(min(len(retrieved), k))])

    def idcg(relevant, k):
        return sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])

    def ndcg_at_k(retrieved, relevant, k):
        dcg_val = dcg(retrieved, relevant, k)
        idcg_val = idcg(relevant, k)
        return dcg_val / idcg_val if idcg_val > 0 else 0.0

    # التقييم لجميع الاستعلامات
    precisions, recalls, maps, ndcgs = [], [], [], []

    for qid, retrieved_docs in tqdm(results.items(), desc="📊 تقييم الاستعلامات"):
        retrieved_doc_ids = [doc_id for doc_id, _ in retrieved_docs]
        relevant_doc_ids = qrels[qid]

        precisions.append(precision_at_k(retrieved_doc_ids, relevant_doc_ids, k))
        recalls.append(recall_at_k(retrieved_doc_ids, relevant_doc_ids, k))
        maps.append(average_precision(retrieved_doc_ids, relevant_doc_ids, k))
        ndcgs.append(ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k))

    # المتوسطات النهائية
    evaluation_results = {
        "Precision@10": round(np.mean(precisions), 4),
        "Recall@10": round(np.mean(recalls), 4),
        "MAP@10": round(np.mean(maps), 4),
        "NDCG@10": round(np.mean(ndcgs), 4),
    }

    return evaluation_results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate_ranking.py <results_json_path>")
    else:
        results_path = sys.argv[1]
        res = evaluate_ranking(results_path)
        print("📈 نتائج التقييم:")
        for metric, value in res.items():
            print(f"{metric}: {value}")
