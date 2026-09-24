from mixedbread_ai_langchain import MixedbreadEmbeddings
from .config import MIXEDBREAD_API_KEY


class EmbeddingGenerator:
    def __init__(self):
        self.embeddings = MixedbreadEmbeddings(
            model="mixedbread-ai/mxbai-embed-large-v1",
            api_key=MIXEDBREAD_API_KEY
        )

    def embed(self, text: str):
        """Embeds a single query."""
        return self.embeddings.embed_query(text)

    def embed_many(self, texts: list, batch_size: int = 128):
        """Embeds documents in safe batches to avoid API limits."""
        clean_texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]

        all_embeddings = []
        total = len(clean_texts)
        for i in range(0, total, batch_size):
            batch = clean_texts[i:i + batch_size]
            print(f"   🔢 Embedding batch {i // batch_size + 1} ({len(batch)} items)...")
            try:
                batch_emb = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_emb)
            except Exception as e:
                print(f"   ⚠️ Batch failed: {e}")
                for t in batch:
                    try:
                        all_embeddings.append(self.embeddings.embed_query(t))
                    except Exception as e2:
                        print(f"   ⚠️ Skipping item: {e2}")
                        all_embeddings.append([0.0] * 1024)
        return all_embeddings