import os
import json
import string
import nltk
from pymongo import MongoClient
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import ir_datasets

# ---------------------- #
#     تحميل الموارد      #
# ---------------------- #
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
punctuations = set(string.punctuation)

def preprocess(text):
    """تنظيف النصوص المتقدم باستخدام Lemmatization"""
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = []
    for token in tokens:
        if token in punctuations:
            continue
        if token in stop_words:
            continue
        if token.isdigit():
            continue
        lemma = lemmatizer.lemmatize(token)
        if len(lemma) < 3:
            continue
        cleaned_tokens.append(lemma)
    return ' '.join(cleaned_tokens)

def process_dataset(dataset_name="beir/quora/test", db_name="ir_project", collection_name="documents_quora_test"):
    # تحميل مجموعة البيانات
    dataset = ir_datasets.load(dataset_name)

    # إعداد MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    collection_docs = db[collection_name]

    # إعداد المجلد
    save_path = os.path.join(dataset_name.replace("/", os.sep), "doc")
    os.makedirs(save_path, exist_ok=True)
    docs_json_path = os.path.join(save_path, "docs.json")

    print(f"🚀 Processing documents from {dataset_name}...")
    bulk_entries = []
    all_docs = []

    for doc in tqdm(dataset.docs_iter(), total=dataset.docs_count()):
        clean_text = preprocess(doc.text)

        bulk_entries.append({
            "doc_id": doc.doc_id,
            "original_text": doc.text,
            "clean_text": clean_text
        })

        all_docs.append({
            "doc_id": doc.doc_id,
            "original_text": doc.text,
            "clean_text": clean_text
        })

    if bulk_entries:
        collection_docs.insert_many(bulk_entries)

    with open(docs_json_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved cleaned documents to: {docs_json_path}")
    print(f"✅ Stored cleaned documents in MongoDB collection: {collection_docs.name}")

# تنفيذ مباشر إذا شغلت كملف مستقل
if __name__ == "__main__":
    process_dataset()
