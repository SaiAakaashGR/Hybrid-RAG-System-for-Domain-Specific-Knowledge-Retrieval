from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection_name = "docs"

count_response = client.count(collection_name=collection_name)
print(f"Total points in collection '{collection_name}': {count_response.count}")
