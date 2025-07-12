from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    dataset: str  # antique or beir
    method: str   # bert or tfidf or hybrid
    top_k: int = 5
