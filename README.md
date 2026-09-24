# 🧠 OmniDoc AI — Chat with Any Document

**OmniDoc AI** is a **Multimodal Retrieval-Augmented Generation (MMRAG)** system that lets you ask questions about PDF documents containing **both text and images** and get accurate, context-aware answers.

Unlike standard chatbots, OmniDoc AI understands **images, charts, and diagrams** inside your documents — not just text.

---

## ✨ Features

- 📄 Extracts text **and** images from PDF documents using **Docling**
- 🧠 Generates semantic summaries of text and images with **Gemini 2.5 Flash**
- 🔢 Creates **1024-dimensional vector embeddings** using **MixedBread AI**
- 🗄️ Stores embeddings in **ChromaDB** for efficient similarity search
- 🎯 Uses **Maximum Marginal Relevance (MMR)** retrieval for diverse results
- 🏆 Re-ranks with **BAAI BGE Cross-Encoder** for higher accuracy
- 💬 Generates grounded answers using **Llama 3** (via Ollama)
- 🌐 Includes a **professional Streamlit web UI** with source citations

---

## 🏗️ System Architecture

```
PDF ──► Docling ──► Text + Images
                     │
                     ├── Gemini: Summaries & Descriptions
                     │
                     ├── MixedBread AI: Vector Embeddings
                     │
                     └── ChromaDB: Vector Store
                              │
User Question ──► MMR Retrieval ──► BGE Reranker
                              │
                              └── Llama 3 ──► Answer with Citations
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| PDF Parsing | Docling |
| Text/Image Summarization | Gemini 2.5 Flash |
| Embeddings | MixedBread AI (`mxbai-embed-large-v1`) |
| Vector Store | ChromaDB |
| Retrieval | Maximum Marginal Relevance (MMR) |
| Re-ranking | BAAI BGE Reranker v2 |
| LLM | Llama 3.2 (via Ollama) |
| Web UI | Streamlit |

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Uma31-Sree/OmniDoc-AI.git
cd OmniDoc-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Then edit .env and add your API keys
```

---

## 🔑 API Keys Required

| Service | Where to Get It |
|---------|-----------------|
| **Gemini API Key** | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| **MixedBread API Key** | [platform.mixedbread.com](https://platform.mixedbread.com) |

**Also install Ollama** for local Llama 3:
- Download from [ollama.com/download](https://ollama.com/download)
- Run `ollama pull llama3.2`

---

## 🚀 Usage

### Option 1: Web UI (Recommended)

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, upload a PDF, and start asking questions.

### Option 2: Terminal

```bash
python main.py your_document.pdf
```

---

## 🎯 How It Works

1. **Parse** — Docling splits your PDF into text chunks and images
2. **Summarize** — Gemini creates semantic summaries of each chunk
3. **Embed** — MixedBread AI converts summaries into 1024-dim vectors
4. **Store** — ChromaDB indexes the vectors for fast similarity search
5. **Retrieve** — MMR fetches diverse candidates, BGE reranks for accuracy
6. **Answer** — Llama 3 generates a grounded answer with source citations

---

## 📸 Screenshots

### Home Screen
![Home Screen](Home Screen.png)

### Chat Interface
![Chat Interface](Chat Interface.png)

### Retrieved Sources
![Retrieved Sources1](Retrieved Sources1.png)
![Retrieved Sources2](Retrieved Sources2.png)


---

## 🔮 Future Enhancements

- [ ] Support for additional formats (DOCX, PPTX, HTML)
- [ ] Multi-document conversational retrieval
- [ ] Real-time document ingestion
- [ ] Hybrid keyword + semantic search
- [ ] Docker containerization

---

## 📜 License

MIT License

---

## 👤 Author

**Uma Sree**
GitHub: [@Uma31-Sree](https://github.com/Uma31-Sree)