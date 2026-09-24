import sys
from backend.document_processing import DocumentProcessor
from backend.summaries import Summarizer
from backend.embeddings import EmbeddingGenerator
from backend.indexing import VectorStore
from backend.retrieval import Retriever
from backend.generation import AnswerGenerator


def index_document(pdf_path: str):
    """Phase 1: Extract, summarize, embed, and store the PDF."""
    print("\n" + "=" * 60)
    print("📄 PHASE 1: INDEXING")
    print("=" * 60)

    print("\n[1/4] Parsing PDF with Docling...")
    processor = DocumentProcessor()
    text_chunks, images = processor.process(pdf_path)
    print(f"      ✅ Extracted {len(text_chunks)} chunks and {len(images)} images")

    if not text_chunks:
        print("❌ No text found in PDF. Exiting.")
        return False

    print("\n[2/4] Summarizing chunks with Gemini...")
    summarizer = Summarizer()
    all_docs = []
    metadatas = []
    for i, chunk in enumerate(text_chunks, 1):
        original, summary = summarizer.summarize_text_pair(chunk)
        all_docs.append(original)
        metadatas.append({"type": "original", "chunk_id": i})
        if summary and summary != original:
            all_docs.append(summary)
            metadatas.append({"type": "summary", "chunk_id": i})
        if i % 5 == 0 or i == len(text_chunks):
            print(f"      ✅ Summarized {i}/{len(text_chunks)} chunks")

    print(f"      → Total docs (original + summary): {len(all_docs)}")

    print("\n[3/4] Generating embeddings with MixedBread...")
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_many(all_docs)
    print(f"      ✅ Generated {len(embeddings)} embeddings")

    print("\n[4/4] Storing in ChromaDB...")
    store = VectorStore()
    store.reset()
    store.add_documents(
        ids=[f"doc_{i}" for i in range(len(all_docs))],
        embeddings=embeddings,
        metadatas=metadatas,
        documents=all_docs
    )
    print(f"      ✅ Total docs in DB: {store.count()}")
    return True


def query_loop():
    """Phase 2: Ask questions interactively."""
    print("\n" + "=" * 60)
    print("💬 PHASE 2: QUERYING")
    print("=" * 60)

    retriever = Retriever()
    embedder = EmbeddingGenerator()
    generator = AnswerGenerator()

    print("\n✅ Ready! Ask questions about your document.")
    print("   Type 'exit' to quit.\n")

    while True:
        try:
            question = input("❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if question.lower() in ("exit", "quit", "q"):
            print("👋 Goodbye!")
            break
        if not question:
            continue

        print("\n🔍 Retrieving relevant chunks...")
        q_emb = embedder.embed(question)
        docs = retriever.mmr_search(q_emb, k=6, fetch_k=30)
        reranked = retriever.rerank(question, docs, top_k=4)

        print(f"   → Retrieved {len(reranked)} chunks")
        print("\n📚 Top retrieved chunk (preview):")
        print(f"   {reranked[0][:200]}...")

        print("\n💭 Generating answer with Llama 3...")
        context = "\n\n".join(reranked)
        answer = generator.generate(context, question)
        print(f"\n💬 Answer: {answer}\n")
        print("-" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        print("Example: python main.py test.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if index_document(pdf_path):
        query_loop()


if __name__ == "__main__":
    main()