import chromadb
from sentence_transformers import CrossEncoder
from .config import CHROMA_PERSIST_DIR, COLLECTION_NAME


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        # Load the reranker (downloads on first run, ~2 GB)
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def mmr_search(self, query_embedding, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5):
        """Fetches candidates from ChromaDB and applies MMR for diverse results."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "distances", "embeddings"]
        )

        docs = results["documents"][0]
        dists = results["distances"][0]
        embs = results["embeddings"][0]

        relevance = [1 - d for d in dists]

        selected_indices = []
        candidate_indices = list(range(len(docs)))

        while len(selected_indices) < k and candidate_indices:
            best_score = -float("inf")
            best_idx = None

            for i in candidate_indices:
                rel = relevance[i]

                if selected_indices:
                    max_sim = max(self._cosine(embs[i], embs[j]) for j in selected_indices)
                    mmr = lambda_mult * rel - (1 - lambda_mult) * max_sim
                else:
                    mmr = rel

                if mmr > best_score:
                    best_score = mmr
                    best_idx = i

            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)

        return [docs[i] for i in selected_indices]

    def _cosine(self, v1, v2):
        """Cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = sum(a * a for a in v1) ** 0.5
        n2 = sum(b * b for b in v2) ** 0.5
        return dot / (n1 * n2 + 1e-9)

    def rerank(self, query: str, documents: list, top_k: int = 3):
        """Re-ranks documents using the BGE Cross-Encoder."""
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]