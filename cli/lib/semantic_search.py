from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re
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

    def search(self, query, limit):
        """
        Search for documents similar to the query using cosine similarity.
        
        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            
        Returns:
            List of dictionaries containing score, title, and description.
            
        Raises:
            ValueError: If embeddings are not loaded.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)
        
        # Calculate cosine similarity between query and all document embeddings
        # Cosine similarity = dot product of normalized vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)
        
        # Create list of (similarity_score, document) tuples
        results = [(similarities[i], self.documents[i]) for i in range(len(self.documents))]
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Return top results with score, title, and description
        return [
            {
                'score': score,
                'title': doc['title'],
                'description': doc['description']
            }
            for score, doc in results[:limit]
        ]


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


def embed_query_text(query):
    """
    Generate and display embedding information for a query.
    
    Args:
        query: The query text to generate an embedding for.
    """
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


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


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class ChunkedSemanticSearch(SemanticSearch):
    DEFAULT_SEMANTIC_CHUNK_SIZE = 4
    DEFAULT_CHUNK_OVERLAP = 1
    
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def semantic_chunk(self, text, max_chunk_size=None, overlap=None):
        """
        Split text into sentence-based chunks with overlap.
        
        Args:
            text: The text to chunk.
            max_chunk_size: Maximum number of sentences per chunk.
            overlap: Number of overlapping sentences between chunks.
            
        Returns:
            List of text chunks.
        """
        if max_chunk_size is None:
            max_chunk_size = self.DEFAULT_SEMANTIC_CHUNK_SIZE
        if overlap is None:
            overlap = self.DEFAULT_CHUNK_OVERLAP
            
        # Split into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        
        # Create chunks with overlap
        chunks = []
        i = 0
        while i < len(sentences):
            # Get the chunk
            chunk_sentences = sentences[i:i + max_chunk_size]
            if not chunk_sentences:
                break
            
            # Skip tiny trailing chunks
            if chunks and len(chunk_sentences) <= overlap:
                break
                
            chunk = " ".join(chunk_sentences)
            chunks.append(chunk)
            
            # Move to next chunk position
            i += max_chunk_size - overlap
            
            # Prevent infinite loop
            if overlap >= max_chunk_size:
                break
        
        return chunks

    def build_chunk_embeddings(self, documents):
        """
        Generate embeddings for document chunks and save to disk.
        
        Args:
            documents: List of document dictionaries (movies).
            
        Returns:
            The generated chunk embeddings.
        """
        # Populate documents and document_map
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        # Create lists for chunks and metadata
        all_chunks = []
        chunk_metadata = []
        
        # Process each document
        for movie_idx, doc in enumerate(documents):
            description = doc.get('description', '').strip()
            
            # Skip if description is empty
            if not description:
                continue
            
            # Use semantic_chunk method to split into chunks
            doc_chunks = self.semantic_chunk(description, max_chunk_size=4, overlap=1)
            
            # Add chunks and metadata
            total_chunks = len(doc_chunks)
            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'movie_idx': movie_idx,
                    'chunk_idx': chunk_idx,
                    'total_chunks': total_chunks
                })
        
        # Generate embeddings for all chunks
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        
        # Save to cache
        os.makedirs('cache', exist_ok=True)
        np.save('cache/chunk_embeddings.npy', self.chunk_embeddings)
        
        with open('cache/chunk_metadata.json', 'w') as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Load chunk embeddings from cache if available, otherwise build them.
        
        Args:
            documents: List of document dictionaries (movies).
            
        Returns:
            The chunk embeddings (either loaded or newly created).
        """
        # Populate documents and document_map
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        # Check if cached embeddings and metadata exist
        embeddings_path = 'cache/chunk_embeddings.npy'
        metadata_path = 'cache/chunk_metadata.json'
        
        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            # Load embeddings
            self.chunk_embeddings = np.load(embeddings_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata_json = json.load(f)
                self.chunk_metadata = metadata_json['chunks']
            
            return self.chunk_embeddings
        
        # Rebuild embeddings if cache doesn't exist
        return self.build_chunk_embeddings(documents)
