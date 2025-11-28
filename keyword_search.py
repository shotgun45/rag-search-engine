import json
from pathlib import Path


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
    results = []
    for movie in movies:
        if query.lower() in movie["title"].lower():
            results.append(movie)
    
    # Sort by ID ascending and limit results
    results = sorted(results, key=lambda x: x["id"])[:max_results]
    return results
