from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        """
        Initialize the SemanticSearch with the all-MiniLM-L6-v2 model.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


def verify_model():
    """
    Create an instance of SemanticSearch and print model information.
    """
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
