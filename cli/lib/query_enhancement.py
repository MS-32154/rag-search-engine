import os
from typing import Optional

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

    response = client.models.generate_content(model=model, contents=prompt)
    rewritten = (response.text or "").strip().strip('"')
    return rewritten if rewritten else query


def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case _:
            return query
