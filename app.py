import streamlit as st
import os
import tempfile
import time
from backend.document_processing import DocumentProcessor
from backend.summaries import Summarizer
from backend.embeddings import EmbeddingGenerator
from backend.indexing import VectorStore
from backend.retrieval import Retriever
from backend.generation import AnswerGenerator

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="OmniDoc AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL THEME
# ============================================================
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp li, .stApp strong, .stApp em, .stApp small {
        color: #1f2937 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #f9fafb !important;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] h3 {
        color: #111827 !important;
        font-weight: 700;
    }
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2.5rem 3rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.15);
    }
    .hero h1 {
        font-size: 2.2rem;
        margin: 0;
        font-weight: 800;
        color: #ffffff !important;
    }
    .hero p {
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
        color: #ffffff !important;
    }
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }
    [data-testid="stFileUploader"] {
        border: 1.5px dashed #c7d2fe !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #4f46e5 !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #1f2937 !important;
        background-color: transparent !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #ffffff !important;
        color: #4f46e5 !important;
        border: 1px solid #4f46e5 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderFile"] {
        background-color: #eef2ff !important;
        border: 1px solid #c7d2fe !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderFile"] * {
        color: #1f2937 !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 14px !important;
        padding: 1rem 1.25rem !important;
        margin: 0.75rem 0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }
    [data-testid="stChatMessage"] * {
        color: #1f2937 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #eef2ff !important;
        border-color: #c7d2fe !important;
    }
    [data-testid="stBottom"], [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background-color: #ffffff !important;
    }
    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        padding: 1rem !important;
        border-radius: 12px !important;
    }
    [data-testid="stMetricValue"] {
        color: #4f46e5 !important;
        font-weight: 700;
    }
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid #e5e7eb !important;
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border: none !important;
        color: #ffffff !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        background: #fafafa !important;
    }
    [data-testid="stExpander"] * {
        color: #1f2937 !important;
    }
    [data-testid="stAlert"] { border-radius: 10px !important; }
    [data-testid="stAlert"] * { color: #1f2937 !important; }
    .feature-card {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 14px !important;
        padding: 1.5rem 1.25rem !important;
        height: 100%;
    }
    .feature-card h4 { margin: 0 0 0.5rem 0; color: #111827 !important; }
    .feature-card p { margin: 0; color: #6b7280 !important; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "indexed": False, "chat_history": [], "pdf_name": None, "stats": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🧠 OmniDoc AI")
    st.caption("Multimodal RAG for PDFs")
    st.markdown("---")

    uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file and st.button("⚡ Index Document", use_container_width=True, type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.status("🔄 Indexing document...", expanded=True) as status:
            try:
                t0 = time.time()

                st.write("📄 **[1/4]** Parsing PDF with Docling...")
                processor = DocumentProcessor()
                text_chunks, images = processor.process(tmp_path)
                st.write(f"   → Extracted **{len(text_chunks)}** chunks, **{len(images)}** images")

                if not text_chunks:
                    st.error("❌ No text found in PDF.")
                    st.stop()

                st.write("🧠 **[2/4]** Summarizing with Gemini...")
                summarizer = Summarizer()
                all_docs = []
                metadatas = []
                progress = st.progress(0)
                for i, chunk in enumerate(text_chunks):
                    original, summary = summarizer.summarize_text_pair(chunk)
                    all_docs.append(original)
                    metadatas.append({"type": "original", "chunk_id": i})
                    if summary and summary != original:
                        all_docs.append(summary)
                        metadatas.append({"type": "summary", "chunk_id": i})
                    if i % 5 == 0 or i == len(text_chunks) - 1:
                        progress.progress((i + 1) / len(text_chunks))

                st.write("🔢 **[3/4]** Generating embeddings...")
                embedder = EmbeddingGenerator()
                embeddings = embedder.embed_many(all_docs)

                st.write("🗄️ **[4/4]** Storing in ChromaDB...")
                store = VectorStore()
                store.reset()
                store.add_documents(
                    ids=[f"doc_{i}" for i in range(len(all_docs))],
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=all_docs
                )

                elapsed = time.time() - t0
                st.session_state.indexed = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.stats = {
                    "chunks": len(text_chunks),
                    "embeddings": len(embeddings),
                    "time": elapsed,
                }
                status.update(label="✅ Document indexed!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.indexed:
        st.success(f"📌 **{st.session_state.pdf_name}**")
        if st.session_state.stats:
            s = st.session_state.stats
            c1, c2 = st.columns(2)
            c1.metric("Chunks", s["chunks"])
            c2.metric("Embeddings", s["embeddings"])
            st.metric("Time", f"{s['time']:.1f}s")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
            st.download_button("💾 Save", chat_text, file_name="omnidoc_chat.txt", use_container_width=True)

    st.markdown("---")
    st.caption("⚙️ Docling • Gemini • MixedBread • ChromaDB • Llama 3")

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🧠 OmniDoc AI</h1>
    <p>Chat with any document — powered by Multimodal RAG</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN AREA
# ============================================================
if not st.session_state.indexed:
    st.info("👈 **Upload a PDF** in the sidebar and click **Index Document** to get started.")

    st.markdown("### ✨ How it works")
    cols = st.columns(4)
    features = [
        ("📄", "Parse", "Docling extracts text and images from your PDF"),
        ("🧠", "Summarize", "Gemini generates semantic summaries"),
        ("🔢", "Embed", "MixedBread AI creates vector embeddings"),
        ("💬", "Answer", "Llama 3 responds with grounded citations"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <h4>{icon} {title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"📚 View {len(msg['sources'])} retrieved sources"):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(f"**Source {i}**")
                            st.caption(src[:400] + ("..." if len(src) > 400 else ""))
                            st.markdown("---")

    question = st.chat_input("💬 Ask something about the document...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🔍 Retrieving..."):
                embedder = EmbeddingGenerator()
                retriever = Retriever()
                q_emb = embedder.embed(question)
                docs = retriever.mmr_search(q_emb, k=6, fetch_k=30)
                reranked = retriever.rerank(question, docs, top_k=4)

            with st.spinner("💭 Generating answer..."):
                context = "\n\n".join(reranked)
                generator = AnswerGenerator()
                answer = generator.generate(context, question)

            st.markdown(answer)
            with st.expander(f"📚 View {len(reranked)} retrieved sources"):
                for i, src in enumerate(reranked, 1):
                    st.markdown(f"**Source {i}**")
                    st.caption(src[:400] + ("..." if len(src) > 400 else ""))
                    st.markdown("---")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": reranked,
        })