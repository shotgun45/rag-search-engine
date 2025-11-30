import json
from pathlib import Path
import string
from nltk.stem import PorterStemmer

BM25_K1 = 1.5


def load_movies(movies_path: str | Path) -> list[dict]:
    """Load movies from JSON file."""
    with open(movies_path, "r") as f:
        data = json.load(f)
    return data["movies"]


def search_movies(movies: list[dict], query: str, max_results: int = 5) -> list[dict]:
    """Search for movies with titles containing the query.
    
    Args:
        movies: List of movie dictionaries
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of matching movies, sorted by ID, limited to max_results
    """

    # Prepare translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    query_clean = query.lower().translate(table)
    query_tokens = [t for t in query_clean.split() if t]

    # Load stop words from file
    stopwords_path = Path(__file__).parent / "data" / "stopwords.txt"
    with open(stopwords_path, "r") as sw_file:
        stopwords = set(sw_file.read().splitlines())

    # Remove stop words from query tokens
    query_tokens = [t for t in query_tokens if t not in stopwords]

    # Stem query tokens
    stemmer = PorterStemmer()
    query_tokens_stemmed = [stemmer.stem(t) for t in query_tokens]

    results = []
    for movie in movies:
        title_clean = movie["title"].lower().translate(table)
        title_tokens = [t for t in title_clean.split() if t]
        # Remove stop words from title tokens
        title_tokens = [t for t in title_tokens if t not in stopwords]
        # Stem title tokens
        title_tokens_stemmed = [stemmer.stem(t) for t in title_tokens]
        # Match if any stemmed query token is a substring of any stemmed title token
        if any(qt in tt for qt in query_tokens_stemmed for tt in title_tokens_stemmed):
            results.append(movie)

    # Sort by ID ascending and limit results
    results = sorted(results, key=lambda x: x["id"])[:max_results]
    return results
