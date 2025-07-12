import ir_datasets
import re
import joblib
import os
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pymongo import MongoClient
import nltk
from nltk.stem import WordNetLemmatizer

def main():
    # تحميل بيانات NLTK الضرورية
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    lemmatizer = WordNetLemmatizer()

    def tokenize(text):
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def lemmatize_tokens(tokens):
        return [lemmatizer.lemmatize(token) for token in tokens]

    def clean_text(text):
        tokens = tokenize(text)
        lemmas = lemmatize_tokens(tokens)
        return ' '.join(lemmas)

    # تحميل dataset
    dataset = ir_datasets.load("antique/test/non-offensive")

    # تحميل موديل BERT
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
        return emb.squeeze().cpu().numpy()

    # إعداد MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ir_project"]
    collection = db["queries_antique_train"]

    query_docs = []
    query_ids = []
    embeddings = []

    print("🔄 Processing queries...")

    for q in tqdm(dataset.queries_iter()):
        cleaned = clean_text(q.text)
        emb = get_bert_embedding(cleaned)
        query_doc = {
            "query_id": q.query_id,
            "original_text": q.text,
            "clean_text": cleaned,
            "bert_embedding": emb.tolist()
        }
        query_docs.append(query_doc)
        query_ids.append(q.query_id)
        embeddings.append(emb)

    if query_docs:
        collection.delete_many({})
        collection.insert_many(query_docs)
        print(f"✅ تم تخزين {len(query_docs)} استعلام في MongoDB في الكوليكشن: {collection.name}")

    output_dir = r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Query Processing\Bertquery\antique\train\query_embeddings"
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump({
        "query_ids": query_ids,
        "embeddings": embeddings,
        "model_name": MODEL_NAME
    }, os.path.join(output_dir, "bert_query_embeddings.joblib"))

    print(f"✅ تم حفظ تمثيلات الاستعلامات في {output_dir}")

if __name__ == "__main__":
    main()
