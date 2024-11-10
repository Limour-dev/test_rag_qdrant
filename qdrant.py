import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from m98_rag import embd, readChunks

QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_KEY = os.getenv('QDRANT_KEY', '')

client = QdrantClient(url=QDRANT_URL,
                      api_key=QDRANT_KEY,
                      timeout=100)

if not client.collection_exists("my_collection"):
    client.create_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

if __name__ == '__main__':
    print(QDRANT_URL)
    print(QDRANT_KEY)
    chunks = readChunks('./test.md')
    for i in range(0, len(chunks), 8):
        batch = chunks[i: i+8]
        print(batch)
        vectors = embd(batch)
        client.upsert(
            collection_name="my_collection",
            points=[
                PointStruct(
                    id=i+idx,
                    vector=vector[1],
                    payload={"text": vector[0]}
                )
                for idx, vector in enumerate(vectors)
            ]
        )
    # 进行搜索
    query_vector = embd(['机器人限拥令是什么？'])
    hits = client.search(
        collection_name="my_collection",
        query_vector=query_vector[0][1],
        limit=5  # Return 5 closest points
    )
    print(hits[0].payload['text'])