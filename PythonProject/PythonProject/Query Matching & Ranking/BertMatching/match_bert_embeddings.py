# match_bert_embeddings.py

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json
import os

def match_bert_embeddings(
    docs_embedding_path: str,
    queries_embedding_path: str,
    output_path: str = "bert_results.json",
    top_k: int = 100,
    batch_size_queries: int = 100
):
    print("📥 تحميل تمثيلات الوثائق...")
    doc_data = joblib.load(docs_embedding_path)
    doc_ids = doc_data["doc_ids"]
    doc_embeddings = np.vstack(doc_data["embeddings_matrix"])  # shape: (num_docs, dim)

    print("📥 تحميل تمثيلات الاستعلامات...")
    query_data = joblib.load(queries_embedding_path)
    query_ids = query_data["query_ids"]
    query_embeddings = np.vstack(query_data["embeddings"])  # shape: (num_queries, dim)

    results = {}

    num_queries = len(query_embeddings)
    print(f"📊 عدد الاستعلامات: {num_queries}, عدد الوثائق: {len(doc_ids)}")

    for start in tqdm(range(0, num_queries, batch_size_queries), desc="🔄 مطابقة دفعات الاستعلامات"):
        end = min(start + batch_size_queries, num_queries)
        batch_queries = query_embeddings[start:end]

        sim_matrix = cosine_similarity(batch_queries, doc_embeddings)

        for i, query_idx in enumerate(range(start, end)):
            sims = sim_matrix[i]
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_scores = sims[top_indices]
            sorted_idx = top_indices[np.argsort(-top_scores)]

            results[query_ids[query_idx]] = [
                (doc_ids[idx], float(sims[idx])) for idx in sorted_idx
            ]

    # حفظ النتائج
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ تم حفظ النتائج في: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Match BERT embeddings between docs and queries")
    parser.add_argument("--docs_embedding_path", required=True, help="Path to document embeddings joblib file")
    parser.add_argument("--queries_embedding_path", required=True, help="Path to queries embeddings joblib file")
    parser.add_argument("--output_path", default="bert_results.json", help="Output json results path")
    parser.add_argument("--top_k", type=int, default=100, help="Top K results to retrieve")
    parser.add_argument("--batch_size_queries", type=int, default=100, help="Batch size for queries processing")

    args = parser.parse_args()

    match_bert_embeddings(
        docs_embedding_path=args.docs_embedding_path,
        queries_embedding_path=args.queries_embedding_path,
        output_path=args.output_path,
        top_k=args.top_k,
        batch_size_queries=args.batch_size_queries,
    )
