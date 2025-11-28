import string
from pathlib import Path
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
        for token in tokens:
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
        Save index and docmap to disk using pickle.
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

    def load(self):
        """
        Load index and docmap from disk using pickle.
        Raises FileNotFoundError if files do not exist.
        """
        import os
        import pickle
        cache_dir = "cache"
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        if not os.path.exists(index_path) or not os.path.exists(docmap_path):
            raise FileNotFoundError("Index or docmap file not found in cache directory.")
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
