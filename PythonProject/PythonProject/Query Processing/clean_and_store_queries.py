import re
import json
import ir_datasets
from pymongo import MongoClient
from nltk import pos_tag, word_tokenize, download
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

def main():
    # تحميل الموارد من NLTK
    download('punkt')
    download('stopwords')
    download('averaged_perceptron_tagger')
    download('wordnet')

    # أدوات التنظيف
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        pos_tags = pos_tag(tokens)
        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
        return ' '.join(lemmatized)

    # تحميل الاستعلامات من BEIR
    dataset = ir_datasets.load("beir/quora/test")
    cleaned_queries = []

    for query in dataset.queries_iter():
        clean_q = clean_text(query.text)
        cleaned_queries.append({
            "query_id": query.query_id,
            "original_text": query.text,
            "clean_text": clean_q
        })

    # حفظ نسخة احتياطية بصيغة JSON
    with open("clean_queries.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_queries, f, indent=2, ensure_ascii=False)

    print(f"✅ تم تنظيف {len(cleaned_queries)} استعلام.")

    # # الاتصال بـ MongoDB
    # client = MongoClient("mongodb://localhost:27017/")
    # db = client["ir_project"]
    # collection = db["queries_quora_test"]
    #
    # collection.delete_many({})
    # collection.insert_many(cleaned_queries)
    #
    # print(f"✅ تم إدخال {len(cleaned_queries)} استعلام إلى MongoDB ضمن مجموعة queries_quora_test.")

if __name__ == "__main__":
    main()
