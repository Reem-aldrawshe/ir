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

# ------------- تحميل الموارد -------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
punctuations = set(string.punctuation)

def preprocess(text):
    """تنظيف النص وإرجاع قائمة كلمات"""
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
    return cleaned_tokens

def process_and_store_documents(dataset_name="antique/train", db_name="ir_project", mongo_collection="documents_test"):
    # تحميل مجموعة البيانات
    dataset = ir_datasets.load(dataset_name)

    # إعداد MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    collection_docs = db[mongo_collection]

    # إعداد المسار
    save_path = os.path.join(dataset_name.replace("/", os.sep), "doc")
    os.makedirs(save_path, exist_ok=True)
    docs_json_path = os.path.join(save_path, "docs.json")

    print(f"🚀 Processing documents from {dataset_name}...")

    all_docs = []

    for doc in tqdm(dataset.docs_iter(), total=dataset.docs_count()):
        clean_tokens = preprocess(doc.text)
        clean_text = ' '.join(clean_tokens)

        # حفظ في MongoDB
        doc_entry = {
            "doc_id": doc.doc_id,
            "original_text": doc.text,
            "clean_text": clean_text
        }
        collection_docs.insert_one(doc_entry)

        # حفظ في ملف JSON
        all_docs.append({
            "doc_id": doc.doc_id,
            "original_text": doc.text,
            "clean_text": clean_text
        })

    with open(docs_json_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved cleaned documents to JSON file: {docs_json_path}")
    print(f"✅ Stored cleaned documents in MongoDB collection: {collection_docs.name}")
