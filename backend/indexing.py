import chromadb
from .config import CHROMA_PERSIST_DIR, COLLECTION_NAME


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def add_documents(self, ids: list, embeddings: list, metadatas: list, documents: list):
        """Stores embeddings + metadata + text in ChromaDB."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"✅ Stored {len(ids)} documents in ChromaDB.")

    def count(self) -> int:
        """Returns the number of stored documents."""
        return self.collection.count()

    def reset(self):
        """Deletes the collection (useful for re-indexing)."""
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        print("🧹 Collection reset.")