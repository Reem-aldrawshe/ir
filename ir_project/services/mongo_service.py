from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["ir_project"]

def get_documents(dataset):
    collection = db["documents_quora_test"] if dataset == "beir" else db["documents_test"]
    docs = collection.find({}, {"doc_id": 1, "original_text": 1})
    return {doc["doc_id"]: doc["original_text"] for doc in docs}
