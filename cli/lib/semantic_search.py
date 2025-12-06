from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        """
        Initialize the SemanticSearch with the all-MiniLM-L6-v2 model.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text):
        """
        Generate an embedding for the input text.
        
        Args:
            text: The input text to generate an embedding for.
            
        Returns:
            The embedding vector for the input text.
            
        Raises:
            ValueError: If the input text is empty or contains only whitespace.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or contain only whitespace")
        
        # encode expects a list and returns a list, we take the first element
        embeddings = self.model.encode([text])
        return embeddings[0]


def verify_model():
    """
    Create an instance of SemanticSearch and print model information.
    """
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text):
    """
    Generate and display embedding information for the input text.
    
    Args:
        text: The input text to generate an embedding for.
    """
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
