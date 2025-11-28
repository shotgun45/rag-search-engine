import string
from pathlib import Path
from collections import Counter
try:
    from nltk.stem import PorterStemmer
except ImportError:
    PorterStemmer = None

class InvertedIndex:
    def __init__(self):
        # Maps token (str) to set of doc IDs (int)
        self.index = {}
        # Maps doc ID (int) to full document object
        self.docmap = {}
        # Maps doc ID (int) to Counter objects for term counts
        self.term_frequencies = {}
        # Prepare translation table to remove punctuation
        self.table = str.maketrans('', '', string.punctuation)
        # Load stop words from file
        stopwords_path = Path(__file__).parent / "data" / "stopwords.txt"
        try:
            with open(stopwords_path, "r") as sw_file:
                self.stopwords = set(sw_file.read().splitlines())
        except Exception:
            self.stopwords = set()
        # Stemmer
        self.stemmer = PorterStemmer() if PorterStemmer else None

    def add_document(self, doc_id, text, doc_obj=None):
        """
        Tokenize text and add tokens to index with doc_id.
        Optionally store the full document object in docmap.
        """
        # Remove punctuation, lowercase, split, remove stopwords, stem
        tokens = [t for t in text.lower().translate(self.table).split() if t]
        tokens = [t for t in tokens if t not in self.stopwords]
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        # Track term frequencies
        if doc_id not in self.term_frequencies:
            from collections import Counter
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
        if doc_obj is not None:
            self.docmap[doc_id] = doc_obj

    def get_documents(self, term):
        """
        Get sorted list of doc IDs for a given token (case-insensitive).
        """
        # Remove punctuation, lowercase, remove stopwords, stem
        term = term.lower().translate(self.table)
        if term in self.stopwords:
            return []
        if self.stemmer:
            term = self.stemmer.stem(term)
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def build(self, movies):
        """
        Build the index from a list of movie dicts.
        Each movie should have 'id', 'title', and 'description'.
        """
        for m in movies:
            doc_id = m['id']
            text = f"{m['title']} {m['description']}"
            self.add_document(doc_id, text, doc_obj=m)

    def save(self):
        """
        Save index, docmap, and term_frequencies to disk using pickle.
        Creates cache directory if it doesn't exist.
        """
        import os
        import pickle
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(cache_dir, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(os.path.join(cache_dir, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        """
        Load index, docmap, and term_frequencies from disk using pickle.
        Raises FileNotFoundError if files do not exist.
        """
        import os
        import pickle
        cache_dir = "cache"
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        tf_path = os.path.join(cache_dir, "term_frequencies.pkl")
        if not (os.path.exists(index_path) and os.path.exists(docmap_path) and os.path.exists(tf_path)):
            raise FileNotFoundError("Index, docmap, or term_frequencies file not found in cache directory.")
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def get_tf(self, doc_id, term):
        """
        Return the frequency of the token in the document with the given ID.
        Tokenize and stem the term. If more than one token, raise an exception.
        """
        tokenized = [t for t in term.lower().translate(self.table).split() if t]
        tokenized = [t for t in tokenized if t not in self.stopwords]
        if self.stemmer:
            tokenized = [self.stemmer.stem(t) for t in tokenized]
        if len(tokenized) != 1:
            raise ValueError("Term must be a single token after tokenization.")
        token = tokenized[0]
        return self.term_frequencies.get(doc_id, {}).get(token, 0)

    def get_idf(self, term):
        """
        Calculate and return the IDF for a given term.
        Uses formula: log((N + 1) / (df + 1))
        """
        import math
        term = term.lower().translate(self.table)
        N = len(self.docmap)
        doc_ids = self.get_documents(term)
        df = len(doc_ids)
        idf = math.log((N + 1) / (df + 1))
        return idf
