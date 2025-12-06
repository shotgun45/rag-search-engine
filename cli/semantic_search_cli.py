#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import lib modules
sys.path.insert(0, str(Path(__file__).parent))

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the model is loaded correctly")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for input text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify movie embeddings")

    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for a query")
    embed_query_parser.add_argument("query", type=str, help="Query text to generate embedding for")

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()