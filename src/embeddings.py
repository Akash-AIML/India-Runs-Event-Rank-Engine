import re
import math
from collections import Counter, defaultdict

def tokenize(text):
    if not text:
        return []
    return re.findall(r'\b[a-z0-9\-]+\b', text.lower())

class BM25Model:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.df = defaultdict(int)
        self.idf = {}
        self.avg_doc_len = 0
        self.num_docs = 0

    def fit(self, tokenized_docs, vocabulary):
        self.num_docs = len(tokenized_docs)
        total_len = sum(len(doc) for doc in tokenized_docs)
        self.avg_doc_len = total_len / self.num_docs if self.num_docs > 0 else 1.0
        
        # Compute Document Frequencies
        for doc in tokenized_docs:
            unique_words = set(doc)
            for w in unique_words:
                self.df[w] += 1
                
        # Compute IDF for vocabulary words
        for w in vocabulary:
            d_f = self.df.get(w, 0)
            self.idf[w] = math.log((self.num_docs - d_f + 0.5) / (d_f + 0.5) + 1.0)

    def score(self, doc, query_words):
        doc_len = len(doc)
        tf = Counter(doc)
        score = 0.0
        for w in query_words:
            if w in tf and w in self.idf:
                term_tf = tf[w]
                score += self.idf[w] * term_tf * (self.k1 + 1) / (term_tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len))
        return score
