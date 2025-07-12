# process_tfidf_both.py

import os
import string
import nltk
import joblib
from pymongo import MongoClient
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
punctuations = set(string.punctuation)

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = []
    for token in tokens:
        if token in punctuations or token in stop_words or token.isdigit():
            continue
        lemma = lemmatizer.lemmatize(token)
        if len(lemma) < 3:
            continue
        cleaned_tokens.append(lemma)
    return ' '.join(cleaned_tokens)

def process_dataset(dataset_name, collection_name):
    print(f"🚀 Processing TF-IDF for: {dataset_name} / {collection_name}")

    client = MongoClient("mongodb://localhost:27017/")
    db = client["ir_project"]
    collection = db[collection_name]

    documents = list(collection.find({}, {"_id": 0, "doc_id": 1, "original_text": 1}))
    documents = [doc for doc in documents if doc.get("original_text", "").strip()]

    if not documents:
        print(f"❌ No documents found in {collection_name}")
        return

    doc_ids = [doc["doc_id"] for doc in documents]
    processed_texts = [preprocess(doc["original_text"]) for doc in tqdm(documents, desc="🧹 Cleaning")]

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        lowercase=False,
        preprocessor=None,
        token_pattern=None,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    vocab_size = len(vectorizer.vocabulary_)
    print(f"✅ Vocabulary size: {vocab_size}")

    save_path = os.path.join(dataset_name.replace("/", os.sep), "doc")
    os.makedirs(save_path, exist_ok=True)
    joblib_path = os.path.join(save_path, "tfidf_data.joblib")

    joblib.dump({
        "tfidf_matrix": tfidf_matrix,
        "vectorizer": vectorizer,
        "doc_ids": doc_ids
    }, joblib_path)

    print(f"✅ Saved TF-IDF to: {joblib_path}")
    print("-" * 60)

if __name__ == "__main__":
    datasets = [
        ("antique/train", "documents_test"),
        ("beir/quora/test", "documents_quora_test"),
    ]

    for dataset_name, collection_name in datasets:
        process_dataset(dataset_name, collection_name)

    print("🎉 All datasets processed successfully!")
