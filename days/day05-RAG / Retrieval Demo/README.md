# 📚 Day 05 — Retrieval-Augmented Generation (RAG) with Document Upload

This project implements a **production-aware RAG (Retrieval-Augmented Generation) system** that allows users to:

- Upload multiple documents (TXT or PDF)
- Ask questions about their content
- Retrieve the most relevant document chunks using vector similarity
- Generate grounded answers using an LLM
- Automatically fallback to a local embedding model if API quota is exceeded
- Cache embeddings locally for efficiency

---

## 🚀 Features

- 🔍 Vector search using FAISS
- 🧠 OpenAI embeddings (with automatic fallback to SentenceTransformers)
- 📂 Multi-file upload (TXT + PDF)
- ⚡ Embedding cache using `shelve`
- 🔁 Exponential retry logic for API robustness
- 🌐 Interactive UI with Gradio
- 📊 Transparent retrieval: shows top-k retrieved snippets

---

## 🏗 Architecture

1. **Document Upload**
   - Accepts multiple `.txt` or `.pdf` files.
   - Extracts raw text from each file.

2. **Embedding Layer**
   - Uses `text-embedding-3-small` (OpenAI) when available.
   - Falls back to `all-MiniLM-L6-v2` (SentenceTransformers) if quota is exceeded.
   - Stores embeddings in local cache.

3. **Vector Index**
   - FAISS `IndexFlatL2` for similarity search.
   - Retrieves top-k most relevant documents.

4. **Generation**
   - Constructs a context-grounded prompt.
   - Generates answer using `gpt-4o-mini`.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/AMZOUG/day05-rag-demo.git
