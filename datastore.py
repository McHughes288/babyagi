import numpy as np
import openai


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]

class DataStore:
    def __init__(self, data=[], embedding_engine="text-embedding-ada-002"):
        self._embedding_engine = embedding_engine
        self._embeddings = {}
        self.load_data(data)

    def load_data(self, data=[]):
        self._data = data
    
    def upsert(self, id, task, result):
        embedding = self.create_embedding(result)
        self._data.append({
            "id": id,
            "task": task,
            "result": result,
            "embedding": embedding
        })

    def create_embedding(self, input_str):
        if input_str in self._embeddings:
            return self._embeddings[input_str]
        else:
            embedding = get_ada_embedding(input_str)
            self._embeddings[input_str] = embedding
            return embedding

    def query(self, query, top_k=2):
        query_embedding = self.create_embedding(query)
        similarities = []
        for item in self._data:
            item_embedding = item["embedding"]
            similarity = np.dot(query_embedding, item_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)
            )
            similarities.append((item, similarity))

        sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
        return [result[0]["result"] for result in sorted_results[:top_k]], sorted_results