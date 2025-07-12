# from fastapi import FastAPI
# from pydantic import BaseModel
# from logic.bert import search_with_bert
# from logic.tfidf import search_with_tfidf
# from logic.hybrid import search_with_hybrid  # تأكد أن الملف اسمه hybrid.py

# app = FastAPI(title="IR Project API")

# from logic.hybrid import initialize_hybrid_data

# initialize_hybrid_data("beir")
# initialize_hybrid_data("antique")

# class QueryRequest(BaseModel):
#     query: str
#     dataset: str  # "antique" or "beir"
#     method: str   # "bert", "tfidf", "hybrid"
#     top_k: int = 10

# @app.post("/search")
# async def search(request: QueryRequest):
#     if request.method == "bert":
#         results = search_with_bert(request.query, request.dataset, request.top_k)
#     elif request.method == "tfidf":
#         results = search_with_tfidf(request.query, request.dataset, request.top_k)
#     elif request.method == "hybrid":
#         results = search_with_hybrid(request.query, request.dataset, request.top_k)
#     else:
#         return {"error": "Invalid method"}

#     return {
#         "query": request.query,
#         "method": request.method,
#         "dataset": request.dataset,
#         "results": results
#     }

# @app.get("/")
# async def root():
#     return {"message": "Server is working ✅"}


from fastapi import FastAPI
from pydantic import BaseModel
from services.tfidf_service import tfidf_search
from services.bert_service import bert_search
from services.hybrid_service import hybrid_search

app = FastAPI(
    title="IR Project API",
    description="🔍 API لاسترجاع المعلومات باستخدام TF-IDF و BERT و Hybrid",
    version="1.0.0"
)

class SearchRequest(BaseModel):
    query: str
    dataset: str  # "antique" أو "beir"
    method: str   # "tfidf" أو "bert" أو "hybrid"
    top_k: int = 10


@app.post("/search")
def search(request: SearchRequest):
    if request.method == "tfidf":
        results = tfidf_search(
            query=request.query,
            dataset=request.dataset,
            top_k=request.top_k
        )
    elif request.method == "bert":
        results = bert_search(
            query=request.query,
            dataset=request.dataset,
            top_k=request.top_k
        )
    elif request.method == "hybrid":
        results = hybrid_search(
            query=request.query,
            dataset=request.dataset,
            top_k=request.top_k
        )
    else:
        return {"error": f"🔴 طريقة البحث {request.method} غير مدعومة"}

    return {
        "query": request.query,
        "dataset": request.dataset,
        "method": request.method,
        "results": results
    }

