from rank_bm25 import BM25Okapi
import re

class BM25Index:

    def __init__(self):
        self.corpus = []
        self.tokenized = []
        self.sources = []
        self.bm25 = None

    def _tokenize(self, text: str):
        text = text.lower()
        return re.findall(r"\w+", text)

    def build(self, contexts: list[str], sources: list[str]):
        self.corpus = contexts
        self.sources = sources
        self.tokenized = [self._tokenize(c) for c in contexts]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int):
        if not self.bm25:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            zip(self.corpus, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [c for c, _ in ranked[:top_k]]
