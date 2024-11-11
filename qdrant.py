import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from m98_rag import embd, readChunks, rerank

QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_KEY = os.getenv('QDRANT_KEY', '')

client = QdrantClient(url=QDRANT_URL,
                      api_key=QDRANT_KEY,
                      timeout=100)


class Hits(list):
    def __init__(self, hits, query_: str, limit_: int):
        super().__init__(hits)
        self.query = query_
        self.limit = limit_
        self.res = None

    def rerank(self):
        if self.res is None:
            hits = [hit.payload['text'] for hit in self]
            self.res = rerank(hits, query_=self.query, top_n_=self.limit)
        return self.res

    def top(self, top_n=3, related_=True):
        res = self.rerank()
        if related_:
            return [hit[0] for hit in res[:top_n] if hit[1] > 0]
        else:
            return [hit[0] for hit in res[:top_n]]

    def print(self):
        return ('\n' + '=' * 50 + '\n').join(self.top())


class RAG:
    def __init__(self, collection_name="my_collection", vectors_size=768):
        self.collection_name = collection_name
        self.vectors_size = vectors_size
        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vectors_size, distance=Distance.COSINE)
            )

    def upsert(self, chunks_: list, ID_: int = 0):
        for i in range(0, len(chunks_), 8):
            batch = chunks_[i: i + 8]
            print(batch)
            vectors = embd(batch)
            client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=ID_ + i + idx,
                        vector=vector[1],
                        payload={"text": vector[0]}
                    )
                    for idx, vector in enumerate(vectors)
                ]
            )

    def size(self):
        count = client.count(collection_name=self.collection_name)
        print(self.collection_name, 'size:', count.count)
        return count.count

    def clear(self):
        client.delete_collection(collection_name=self.collection_name)
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vectors_size, distance=Distance.COSINE)
        )

    def search(self, query_: str, limit_: int = 5):
        query_vector = embd([query_])
        hits = client.search(
            collection_name=self.collection_name,
            query_vector=query_vector[0][1],
            limit=limit_  # Return 5 closest points
        )
        return Hits(hits, query_, limit_)


if __name__ == '__main__':
    print(QDRANT_URL)
    print(QDRANT_KEY)
    test = RAG(vectors_size=1536)
    if test.size() <= 0:
        chunks = readChunks('./test.md')
        test.upsert(chunks)
    tmp = test.search('机器人限拥令是什么')
    print(tmp.print())
