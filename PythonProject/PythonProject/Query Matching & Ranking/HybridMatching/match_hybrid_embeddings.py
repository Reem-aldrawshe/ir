import joblib
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import json

def match_hybrid_embeddings(
    queries_path: str,
    hybrid_chunks_dir: str,
    output_path: str,
    top_k: int = 100,
    alpha: float = 0.5
):
    print("🔹 تحميل الاستعلامات الهجينة...")
    queries_data = joblib.load(queries_path)

    query_ids = queries_data["query_ids"]
    tfidf_indices_list = queries_data["tfidf_indices"]
    tfidf_values_list = queries_data["tfidf_values"]
    bert_queries = np.array(queries_data["bert_embeddings"], dtype=np.float32)

    num_queries = len(query_ids)

    # إعادة بناء مصفوفة TF-IDF من القوائم
    indptr = [0]
    indices = []
    data = []

    for i in range(num_queries):
        indices.extend(tfidf_indices_list[i])
        data.extend(tfidf_values_list[i])
        indptr.append(len(indices))

    vocab_size = 102029
    tfidf_query_matrix = csr_matrix((data, indices, indptr), shape=(num_queries, vocab_size))

    print(f"📊 عدد الاستعلامات: {num_queries}, حجم مفردات TF-IDF: {tfidf_query_matrix.shape[1]}")

    results = {qid: [] for qid in query_ids}
    chunk_files = [f for f in os.listdir(hybrid_chunks_dir) if f.endswith(".joblib")]
    chunk_files.sort()

    for chunk_file in tqdm(chunk_files, desc="🧩 معالجة Chunks"):
        chunk_path = os.path.join(hybrid_chunks_dir, chunk_file)
        chunk_data = joblib.load(chunk_path)

        tfidf_docs = chunk_data["tfidf_chunk"]       # sparse matrix
        bert_docs = np.array(chunk_data["bert_chunk"], dtype=np.float32)
        doc_ids = chunk_data["doc_ids"]

        if tfidf_query_matrix.shape[1] != tfidf_docs.shape[1]:
            raise ValueError(
                f"❌ عدم تطابق بين أبعاد TF-IDF للاستعلامات ({tfidf_query_matrix.shape[1]}) "
                f"والوثائق ({tfidf_docs.shape[1]}) في {chunk_file}"
            )

        sim_tfidf = cosine_similarity(tfidf_query_matrix, tfidf_docs)
        sim_bert = cosine_similarity(bert_queries, bert_docs)
        sim_hybrid = alpha * sim_tfidf + (1 - alpha) * sim_bert

        for i, qid in enumerate(query_ids):
            sims = sim_hybrid[i]
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_scores = sims[top_indices]
            sorted_idx = top_indices[np.argsort(-top_scores)]

            results[qid].extend([(doc_ids[idx], float(sims[idx])) for idx in sorted_idx])

    # ترتيب وأخذ أعلى top_k لكل استعلام
    final_results = {}
    for qid, docs in results.items():
        docs.sort(key=lambda x: -x[1])
        final_results[qid] = docs[:top_k]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"✅ تم حفظ النتائج في: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="مطابقة الاستعلامات الهجينة مع المستندات")
    parser.add_argument("--queries_path", type=str, required=True, help="مسار ملف الاستعلامات الهجينة (joblib)")
    parser.add_argument("--hybrid_chunks_dir", type=str, required=True, help="مجلد ملفات chunks الهجينة")
    parser.add_argument("--output_path", type=str, required=True, help="مسار حفظ النتائج (json)")
    parser.add_argument("--top_k", type=int, default=100, help="عدد أعلى النتائج المراد إرجاعها")
    parser.add_argument("--alpha", type=float, default=0.5, help="وزن TF-IDF في المزج (بين 0 و 1)")

    args = parser.parse_args()

    match_hybrid_embeddings(
        queries_path=args.queries_path,
        hybrid_chunks_dir=args.hybrid_chunks_dir,
        output_path=args.output_path,
        top_k=args.top_k,
        alpha=args.alpha
    )
