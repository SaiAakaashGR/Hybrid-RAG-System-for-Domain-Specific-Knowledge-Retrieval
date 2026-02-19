import logging
from data_loader import embed_texts
from vector_db import QdrantStorage
from reranker import Reranker
from bm25_index import BM25Index

logger = logging.getLogger("rag")


class RetrievalPipeline:

    def __init__(self):
        self.store = QdrantStorage()

        # ---------- Load reranker ----------
        try:
            self.reranker = Reranker()
            self.reranker_available = True
            logger.info("Reranker loaded")
        except Exception as e:
            logger.warning(f"Reranker unavailable: {e}")
            self.reranker_available = False

        # ---------- Build BM25 ----------
        try:
            contexts, sources = self.store.get_all_texts()
            self.bm25 = BM25Index()
            self.bm25.build(contexts, sources)
            self.bm25_available = True
            logger.info("BM25 index ready")
        except Exception as e:
            logger.warning(f"BM25 unavailable: {e}")
            self.bm25_available = False

    def retrieve(self, question: str, top_k: int = 5):

        query_vec = embed_texts([question])[0]

        # -------- VECTOR SEARCH --------
        vector_k = max(top_k * 4, 20)
        vec_found = self.store.search(query_vec, vector_k)
        vector_contexts = vec_found["contexts"]
        sources = vec_found["sources"]

        # -------- BM25 SEARCH --------
        bm25_contexts = []
        if self.bm25_available:
            bm25_contexts = self.bm25.search(question, vector_k)

        # -------- MERGE --------
        merged = list(dict.fromkeys(vector_contexts + bm25_contexts))

        logger.info(
            f"Vector:{len(vector_contexts)} BM25:{len(bm25_contexts)} merged:{len(merged)}"
        )

        # -------- RERANK --------
        if self.reranker_available:
            try:
                best = self.reranker.rerank(
                    question,
                    merged,
                    top_k
                )
                return best, sources, "hybrid_rerank"
            except Exception as e:
                logger.error(f"Rerank failed: {e}")

        # -------- FALLBACK --------
        return merged[:top_k], sources, "hybrid_no_rerank"
