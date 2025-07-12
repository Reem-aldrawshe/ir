import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json
from scipy.sparse import csr_matrix

def match_tfidf_with_joblib(
    docs_joblib_path: str,
    queries_joblib_path: str,
    output_path: str = "tfidf_results_batch.json",
    top_k: int = 100,
    batch_size_queries: int = 100
):
    print("🔹 تحميل ملف الوثائق (tfidf_matrix + doc_ids)...")
    docs_data = joblib.load(docs_joblib_path)
    doc_ids = docs_data["doc_ids"]
    tfidf_docs: csr_matrix = docs_data["tfidf_matrix"]

    print("🔹 تحميل ملف الاستعلامات (query_tfidf_matrix + query_ids)...")
    queries_data = joblib.load(queries_joblib_path)

    # اختيار مصفوفة الـ tfidf الخاصة بالاستعلامات بطريقة سليمة
    if "query_tfidf_matrix" in queries_data:
        tfidf_queries = queries_data["query_tfidf_matrix"]
    elif "tfidf_matrix" in queries_data:
        tfidf_queries = queries_data["tfidf_matrix"]
    elif "queries_tfidf_matrix" in queries_data:
        tfidf_queries = queries_data["queries_tfidf_matrix"]
    else:
        raise ValueError("❌ لم يتم العثور على أي مصفوفة tfidf في ملف الاستعلامات.")

    # اختيار الـ query_ids بطريقة سليمة
    if "query_ids" in queries_data:
        query_ids = queries_data["query_ids"]
    elif "doc_ids" in queries_data:
        query_ids = queries_data["doc_ids"]
    else:
        raise ValueError("❌ لم يتم العثور على 'query_ids' أو 'doc_ids' في ملف الاستعلامات.")

    num_queries = tfidf_queries.shape[0]
    print(f"📏 عدد الاستعلامات: {num_queries}, عدد الوثائق: {tfidf_docs.shape[0]}")

    results = {}

    for start in tqdm(range(0, num_queries, batch_size_queries), desc="🔄 مطابقة دفعات الاستعلامات"):
        end = min(start + batch_size_queries, num_queries)
        batch_queries = tfidf_queries[start:end]

        # حساب مصفوفة التشابه (batch_size_queries x num_docs)
        sim_matrix = cosine_similarity(batch_queries, tfidf_docs)

        for i, query_idx in enumerate(range(start, end)):
            sims = sim_matrix[i]
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_scores = sims[top_indices]
            sorted_idx = top_indices[np.argsort(-top_scores)]

            results[query_ids[query_idx]] = [
                (doc_ids[idx], float(sims[idx])) for idx in sorted_idx
            ]

    # حفظ النتائج
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ تم حفظ نتائج المطابقة في {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="مطابقة الاستعلامات مع الوثائق باستخدام TF-IDF")
    parser.add_argument("--docs_joblib_path", type=str, required=True, help="مسار ملف الوثائق (joblib)")
    parser.add_argument("--queries_joblib_path", type=str, required=True, help="مسار ملف الاستعلامات (joblib)")
    parser.add_argument("--output_path", type=str, default="tfidf_results_batch.json", help="مسار حفظ النتائج (json)")
    parser.add_argument("--top_k", type=int, default=100, help="عدد أعلى النتائج المراد إرجاعها")
    parser.add_argument("--batch_size_queries", type=int, default=100, help="حجم دفعة الاستعلامات")

    args = parser.parse_args()

    match_tfidf_with_joblib(
        docs_joblib_path=args.docs_joblib_path,
        queries_joblib_path=args.queries_joblib_path,
        output_path=args.output_path,
        top_k=args.top_k,
        batch_size_queries=args.batch_size_queries
    )
