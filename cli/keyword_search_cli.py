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
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Error: Inverted index not found. Please run the build command first.")
                sys.exit(1)
            # Tokenize query (same logic as index)
            import string
            from nltk.stem import PorterStemmer
            table = str.maketrans('', '', string.punctuation)
            stemmer = PorterStemmer()
            tokens = [t for t in args.query.lower().translate(table).split() if t]
            # Remove stop words
            stopwords_path = Path(__file__).parent.parent / "data" / "stopwords.txt"
            try:
                with open(stopwords_path, "r") as sw_file:
                    stopwords = set(sw_file.read().splitlines())
            except Exception:
                stopwords = set()
            tokens = [t for t in tokens if t not in stopwords]
            tokens = [stemmer.stem(t) for t in tokens]
            # Collect up to 5 unique matching doc IDs
            found = set()
            for token in tokens:
                for doc_id in index.get_documents(token):
                    if doc_id not in found:
                        found.add(doc_id)
                        if len(found) == 5:
                            break
                if len(found) == 5:
                    break
            if not found:
                print("No documents found for this query.")
                return
            # Print results
            for idx, doc_id in enumerate(sorted(found), 1):
                movie = index.docmap[doc_id]
                print(f"{idx}. {movie['title']} (ID: {doc_id})")
        case "build":
            # Load movies data
            movies_path = Path(__file__).parent.parent / "data" / "movies.json"
            movies = load_movies(movies_path)
            # Build inverted index
            index = InvertedIndex()
            index.build(movies)
            index.save()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()