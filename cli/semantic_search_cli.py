#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import lib modules
sys.path.insert(0, str(Path(__file__).parent))

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch
import json

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the model is loaded correctly")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for input text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify movie embeddings")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for a query")
    embed_query_parser.add_argument("query", type=str, help="Query text to generate embedding for")

    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk (default: 200)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks (default: 0)")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            # Create SemanticSearch instance
            search = SemanticSearch()
            
            # Load movies from JSON
            movies_path = Path(__file__).parent.parent / "data" / "movies.json"
            with open(movies_path, "r") as f:
                data = json.load(f)
            documents = data["movies"]
            
            # Load or create embeddings
            search.load_or_create_embeddings(documents)
            
            # Perform search
            results = search.search(args.query, args.limit)
            
            # Print results
            for idx, result in enumerate(results, 1):
                print(f"{idx}. {result['title']} (score: {result['score']:.4f})")
                # Truncate description to ~100 chars for display
                desc = result['description']
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                print(f"   {desc}")
                print()
        case "chunk":
            # Split text into words
            words = args.text.split()
            
            # Create chunks with overlap
            chunks = []
            i = 0
            while i < len(words):
                chunk = " ".join(words[i:i + args.chunk_size])
                chunks.append(chunk)
                i += args.chunk_size - args.overlap
                
                # Prevent infinite loop if overlap >= chunk_size
                if args.overlap >= args.chunk_size:
                    break
            
            # Print results
            print(f"Chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, 1):
                print(f"{idx}. {chunk}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()