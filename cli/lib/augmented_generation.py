import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
    load_movies,
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def generate_answer(search_results, query, limit=5):
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Hoopla is a streaming service for movies. You are a RAG agent that provides a human answer
to the user's query based on the documents that were retrieved during search. Provide a comprehensive
answer that addresses the user's query.
a

Query: {query}

Documents:
{context}
"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def rag(query, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = generate_answer(search_results, query, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "answer": answer,
    }


def rag_command(query):
    return rag(query)
