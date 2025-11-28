#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import keyword_search module
sys.path.insert(0, str(Path(__file__).parent.parent))

from keyword_search import load_movies, search_movies
from inverted_index import InvertedIndex



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build and save inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            # Load movies data
            movies_path = Path(__file__).parent.parent / "data" / "movies.json"
            movies = load_movies(movies_path)
            
            # Search for matching movies
            results = search_movies(movies, args.query)
            
            # Print results
            for idx, movie in enumerate(results, 1):
                print(f"{idx}. {movie['title']}")
        case "build":
            # Load movies data
            movies_path = Path(__file__).parent.parent / "data" / "movies.json"
            movies = load_movies(movies_path)
            
            # Build inverted index
            index = InvertedIndex()
            index.build(movies)
            index.save()
            docs = index.get_documents('merida')
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()