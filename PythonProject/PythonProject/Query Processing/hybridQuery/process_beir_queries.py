import os
import re
import joblib
import ir_datasets
import nltk
from tqdm import tqdm
from pymongo import MongoClient
from nltk.stem import WordNetLemmatizer

def main():
    def tokenize(text):
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def lemmatize_tokens(tokens):
        return [lemmatizer.lemmatize(token) for token in tokens]

    def clean_text(text):
        tokens = tokenize(text)
        lemmas = lemmatize_tokens(tokens)
        return " ".join(lemmas)

    nltk.download("wordnet")
    nltk.download("omw-1.4")

    lemmatizer = WordNetLemmatizer()

    # مسارات الملفات
    bert_path = r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Query Processing\Bertquery\BEIR\quora\test\query_embeddings\bert_query_embeddings.joblib"
    vectorizer_path = r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Representation\TF-IDF\beir\quora\test\doc\tfidf_data.joblib"

    # تحميل ملفات BERT و TF-IDF vectorizer الخاص بالوثائق
    bert_data = joblib.load(bert_path)
    vectorizer_data = joblib.load(vectorizer_path)
    vectorizer = vectorizer_data["vectorizer"]

    # تحميل الاستعلامات
    dataset = ir_datasets.load("beir/quora/test")
    query_ids = []
    original_texts = []
    clean_texts = []

    print("🧼 تنظيف الاستعلامات...")
    for query in tqdm(dataset.queries_iter()):
        cleaned = clean_text(query.text)
        if cleaned.strip():
            query_ids.append(query.query_id)
            original_texts.append(query.text)
            clean_texts.append(cleaned)

    print("🔢 تحويل الاستعلامات إلى TF-IDF باستخدام vectorizer الوثائق...")
    tfidf_matrix = vectorizer.transform(clean_texts)

    tfidf_indices_list = []
    tfidf_values_list = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(i).tocoo()
        tfidf_indices_list.append(row.col.tolist())
        tfidf_values_list.append(row.data.tolist())

    bert_embeddings = bert_data["embeddings"]
    bert_model_name = bert_data["model_name"]

    # # تخزين في MongoDB
    # client = MongoClient("mongodb://localhost:27017/")
    # db = client["ir_project"]
    # collection = db["queries_quora_test_hybrid_updated"]
    # collection.delete_many({})
    #
    # query_docs = []
    # for i in tqdm(range(len(query_ids)), desc="Mongo Insert"):
    #     doc = {
    #         "query_id": query_ids[i],
    #         "original_text": original_texts[i],
    #         "clean_text": clean_texts[i],
    #         "bert_embedding": bert_embeddings[i].tolist(),
    #         "tfidf_indices": tfidf_indices_list[i],
    #         "tfidf_values": tfidf_values_list[i],
    #     }
    #     query_docs.append(doc)
    #
    # collection.insert_many(query_docs)
    # print(f"✅ تم تخزين {len(query_docs)} استعلام هجين في MongoDB داخل: {collection.name}")

    # حفظ بصيغة joblib
    output_path = r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Query Processing\hybridQuery\BEIR\quora\test\hybird_query_data.joblib"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    joblib.dump({
        "query_ids": query_ids,
        "original_texts": original_texts,
        "clean_texts": clean_texts,
        "bert_embeddings": bert_embeddings,
        "tfidf_indices": tfidf_indices_list,
        "tfidf_values": tfidf_values_list,
        "bert_model_name": bert_model_name
    }, output_path)

    print(f"📦 تم حفظ تمثيل الاستعلامات الهجين في: {output_path}")

if __name__ == "__main__":
    main()
