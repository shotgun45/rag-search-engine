import argparse


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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.scores:
                normalized = normalize_scores(args.scores)
                for score in normalized:
                    print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()