# build_inverted_index.py

import json
import re
from collections import defaultdict

def tokenize(text):
    # تقسيم النص إلى كلمات بسيطة (يمكن تحسينها لاحقًا بإزالة علامات الترقيم والكلمات التوقف)
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def build_inverted_index(docs):
    inverted_index = defaultdict(set)  # الكلمة -> مجموعة معرفات المستندات

    for doc in docs:
        doc_id = doc["doc_id"]
        text = doc["clean_text"]
        tokens = tokenize(text)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            inverted_index[token].add(doc_id)

    # تحويل المجموعات لقوائم (لتسهيل التخزين أو المعالجة)
    inverted_index = {term: list(doc_ids) for term, doc_ids in inverted_index.items()}
    return inverted_index

def main():
    # قراءة الملف JSON
    path = r"C:\Users\Reem Darawsheh\Desktop\PythonProject\PythonProject\Data Pre-Processing\antique\quora\test\doc\docs.json"
    with open(path, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    print(f"عدد المستندات: {len(docs)}")

    # بناء الفهرس
    inverted_index = build_inverted_index(docs)

    # مثال: عرض بعض المصطلحات وعدد مستنداتها
    for term in list(inverted_index.keys())[:10]:
        print(f"'{term}': {len(inverted_index[term])} مستندات")

    # حفظ الفهرس إلى ملف JSON
    with open('antique_inverted_index.json', 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=2)
    print("✅ تم حفظ الفهرس في antique_inverted_index.json")

if __name__ == "__main__":
    main()
