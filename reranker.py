from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        # Cross-encoder evaluates (query, chunk) pairs
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query: str, contexts: list[str], top_k: int):
        pairs = [(query, c) for c in contexts]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(contexts, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked_contexts = [c for c, _ in ranked[:top_k]]
        return reranked_contexts
