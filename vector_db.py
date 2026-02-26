from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import hashlib

class QdrantStorage: 
    # def __init__(self, url="http://localhost:6333", collection="docs", dim=1536):
    #     self.client = QdrantClient(url=url, timeout=30)
    #     self.collection = collection

    #     existing = {
    #         c.name for c in self.client.get_collections().collections
    #     }

    #     # if not self.client.collection_exists(self.collection):
    #     #     self.client.create_collection(
    #     #         collection_name=collection,
    #     #         vectors_config=VectorParams(size=dim, distance=Distance.COSINE),

    #     #     )
    #     if self.collection not in existing:
    #         self.client.create_collection(
    #             collection_name=self.collection,
    #             vectors_config=VectorParams(
    #                 size=dim,
    #                 distance=Distance.COSINE,
    #             ),
    #         )

    def __init__(self, source_id: str,
                 url="http://localhost:6333",
                 dim=1536):

        self.client = QdrantClient(url=url, timeout=30)

        # ---- collection per document ----
        doc_hash = hashlib.md5(source_id.encode()).hexdigest()[:10]
        self.collection = f"docs_{doc_hash}"

        existing = {
            c.name for c in self.client.get_collections().collections
        }

        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )
    
    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids)) ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k:int = 5):
        results = self.client.search(
            collection_name = self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts=[]
        sources=set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
            if source:
                sources.add(source)
        return {"contexts": contexts, "sources": list(sources)}
    
    def get_all_texts(self):
        scroll = self.client.scroll(
            collection_name=self.collection,
            with_payload=True,
            limit=100000
        )

        contexts = []
        sources = []

        for point in scroll[0]:
            payload = point.payload or {}
            text = payload.get("text")
            source = payload.get("source")

            if text:
                contexts.append(text)
                sources.append(source)

        return contexts, sources

    def collection_exists(self):
        existing = {
            c.name for c in self.client.get_collections().collections
        }
        return self.collection in existing

    def reset_collection(self):
        if self.collection_exists():
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE,
            ),
        )
    def delete_old_collections(self, keep_last=10):
        cols = self.client.get_collections().collections
        doc_cols = [c.name for c in cols if c.name.startswith("docs_")]

        if len(doc_cols) > keep_last:
            for c in sorted(doc_cols)[:-keep_last]:
                self.client.delete_collection(c)


