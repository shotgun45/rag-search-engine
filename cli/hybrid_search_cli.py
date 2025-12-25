#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import lib modules
sys.path.insert(0, str(Path(__file__).parent))

from lib.hybrid_search import HybridSearch


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores using min-max normalization.
    
    Args:
        scores: List of scores to normalize
        
    Returns:
        List of normalized scores in range [0, 1]
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    # If all scores are the same, return list of 1.0 values
    if min_score == max_score:
        return [1.0] * len(scores)
    
    # Min-max normalization: (x - min) / (max - min)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add normalize subcommand
    normalize_parser = subparsers.add_parser(
        "normalize", 
        help="Normalize a list of scores using min-max normalization"
    )
    normalize_parser.add_argument(
        "scores",
        nargs="*",
        type=float,
        help="List of scores to normalize"
    )
    
    # Add weighted-search subcommand
    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Perform weighted hybrid search combining BM25 and semantic search"
    )
    weighted_search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 scores (default: 0.5)"
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.scores:
                normalized = normalize_scores(args.scores)
                for score in normalized:
                    print(f"* {score:.4f}")
        case "weighted-search":
            # Load movies
            movies_path = Path(__file__).parent.parent / "data" / "movies.json"
            with open(movies_path, "r") as f:
                data = json.load(f)
            documents = data["movies"]
            
            # Initialize hybrid search
            hybrid_search = HybridSearch(documents)
            
            # Perform weighted search
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            
            # Print results
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"   BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                # Truncate description to first 100 characters
                description = result['description'][:100]
                if len(result['description']) > 100:
                    description += "..."
                print(f"   {description}")
                if i < len(results):
                    print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()