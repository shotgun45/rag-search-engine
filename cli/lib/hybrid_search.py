import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from semantic_search import ChunkedSemanticSearch
from inverted_index import InvertedIndex


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        try:
            self.idx.load()
        except FileNotFoundError:
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        # Get BM25 results (500x limit to ensure enough results)
        bm25_results = self._bm25_search(query, limit * 500)
        
        # Get semantic results (500x limit to ensure enough results)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        
        # Normalize scores
        bm25_scores = [score for doc_id, score in bm25_results]
        semantic_scores = [result['score'] for result in semantic_results]
        
        normalized_bm25 = self._normalize_scores(bm25_scores)
        normalized_semantic = self._normalize_scores(semantic_scores)
        
        # Create a dictionary mapping document IDs to their scores and document
        doc_scores = {}
        
        # Add BM25 results
        for i, (doc_id, score) in enumerate(bm25_results):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'document': self.idx.docmap[doc_id],
                    'bm25_score': normalized_bm25[i],
                    'semantic_score': 0.0
                }
            else:
                doc_scores[doc_id]['bm25_score'] = normalized_bm25[i]
        
        # Add semantic results
        for i, result in enumerate(semantic_results):
            doc_id = result['id']
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'document': self.semantic_search.document_map[doc_id],
                    'bm25_score': 0.0,
                    'semantic_score': normalized_semantic[i]
                }
            else:
                doc_scores[doc_id]['semantic_score'] = normalized_semantic[i]
        
        # Calculate hybrid scores
        for doc_id in doc_scores:
            bm25 = doc_scores[doc_id]['bm25_score']
            semantic = doc_scores[doc_id]['semantic_score']
            hybrid = alpha * bm25 + (1 - alpha) * semantic
            doc_scores[doc_id]['hybrid_score'] = hybrid
        
        # Sort by hybrid score descending
        sorted_results = sorted(
            doc_scores.items(),
            key=lambda x: x[1]['hybrid_score'],
            reverse=True
        )
        
        # Return top limit results
        return [
            {
                'id': doc_id,
                'title': data['document']['title'],
                'description': data['document']['description'],
                'hybrid_score': data['hybrid_score'],
                'bm25_score': data['bm25_score'],
                'semantic_score': data['semantic_score']
            }
            for doc_id, data in sorted_results[:limit]
        ]
    
    def _normalize_scores(self, scores):
        """Normalize scores using min-max normalization."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        # If all scores are the same, return list of 1.0 values
        if min_score == max_score:
            return [1.0] * len(scores)
        
        # Min-max normalization: (x - min) / (max - min)
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

