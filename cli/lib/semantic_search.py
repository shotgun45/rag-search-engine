from sentence_transformers import SentenceTransformer
import numpy as np
import os
from pathlib import Path


class SemanticSearch:
    def __init__(self):
        """
        Initialize the SemanticSearch with the all-MiniLM-L6-v2 model.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

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

    def build_embeddings(self, documents):
        """
        Generate embeddings for all documents and save to disk.
        
        Args:
            documents: List of document dictionaries (movies).
            
        Returns:
            The generated embeddings.
        """
        self.documents = documents
        
        # Build document_map
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        # Create string representations of documents
        doc_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        
        # Generate embeddings with progress bar
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        
        # Save embeddings to cache
        os.makedirs('cache', exist_ok=True)
        np.save('cache/movie_embeddings.npy', self.embeddings)
        
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        """
        Load embeddings from cache if available, otherwise build them.
        
        Args:
            documents: List of document dictionaries (movies).
            
        Returns:
            The embeddings (either loaded or newly created).
        """
        # Populate documents and document_map
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        # Check if cached embeddings exist
        cache_path = 'cache/movie_embeddings.npy'
        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            # Verify the embeddings match the documents
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        # Rebuild embeddings if cache doesn't exist or doesn't match
        return self.build_embeddings(documents)


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


def verify_embeddings():
    """
    Load or create embeddings for all movies and display information.
    """
    import json
    import sys
    
    search = SemanticSearch()
    
    # Load movies from JSON
    movies_path = Path(__file__).parent.parent.parent / "data" / "movies.json"
    with open(movies_path, "r") as f:
        data = json.load(f)
    documents = data["movies"]
    
    # Load or create embeddings
    embeddings = search.load_or_create_embeddings(documents)
    
    # Print verification info
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
