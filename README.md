# Hybrid-RAG-System-for-Domain-Specific-Knowledge-Retrieval
Production-grade Retrieval-Augmented Generation with Hybrid Search, Reranking, Query Rewriting & Multi-Hop Retrieval

## 🚀 Overview

This project implements a full RAG pipeline for domain-specific PDFs and documents, covering the complete workflow:
1. PDF ingestion & chunking – extract text from uploaded PDFs and split into retrievable chunks.
2. Vector embeddings & storage – generate embeddings using OpenAI embeddings, stored in Qdrant.
3. Hybrid retrieval – combines BM25 sparse search + dense vector search.
4. Cross-encoder reranking – selects the most relevant chunks.
5. Query rewriting – automatically reformulates ambiguous questions.
6. Multi-hop retrieval – expands context iteratively if more evidence is needed.
7. LLM-based answer generation – produces grounded answers from retrieved chunks.
8. Graceful degradation – fallback retrieval if advanced components fail.
9. Evaluation-ready architecture – designed to measure retrieval and answer accuracy.

## 🎯 Why This Project Matters
Most RAG implementations fail because:  
Vector similarity alone misses keyword-critical matches  
Retrieved chunks are poorly ranked  
Complex queries require reasoning across documents  
Systems collapse when embedding search fails  
  
This project solves those limitations using a layered retrieval architecture.  
  
## 🧠 System Architecture
```
            User Query
                │
                ▼
            Query Rewriter (LLM)
                │
                ▼
──────── Hybrid Retrieval ────────
│                                  │
│   BM25 Sparse Search             │
│   (keyword precision)            │
│                                  │
│   Vector Search (Qdrant)         │
│   (semantic similarity)          │
└──────────────┬───────────────────┘
               ▼
        Candidate Pool
               │
               ▼
        Cross-Encoder
           Reranker
               │
               ▼
        Multi-Hop Retrieval
      (context expansion)
               │
               ▼
        Final Context
               │
               ▼
            LLM Answer
```
## ✨ Key Features
✅ PDF Ingestion
Upload PDFs via Streamlit or API.  
Automatic chunking into 1k-token overlapping segments.  
Assigns unique source_id for each document.  
   
✅ Vector Embeddings & Storage  
Uses OpenAI text embeddings (text-embedding-3-small).  
Efficient storage in Qdrant for semantic search.  

✅ Hybrid Retrieval  
Combines BM25 lexical search with vector similarity.  
Improves precision & recall across varied queries.  
  
✅ Neural Reranking  
Cross-encoder selects the most relevant chunks from candidate pool.  
  
✅ Automatic Query Rewriting  
Reformulates vague or underspecified questions for better retrieval.  
  
✅ Multi-Hop Retrieval  
Iteratively fetches additional chunks when more evidence is needed.  
  
✅ Graceful Degradation  
Secondary vector-only or BM25-only retrieval ensures uptime if primary pipeline fails.  
  
✅ Modular, Production-Ready  
Each stage is independent for testing, scaling, or swapping models.  
  
## 🧱 Project Structure
```
RAG/
│
├── main.py                # Entry point
├── retrieval_pipeline.py  # Full retrieval orchestration
├── query_engine.py        # Query execution logic
├── reranker.py            # Cross-encoder reranking
├── bm25_index.py          # Sparse retrieval
├── vector_db.py           # Qdrant interface
├── data_loader.py         # Index builder
├── streamlit_app.py       # UI demo
│
├── qdrant_storage/        # Local vector DB (ignored in git)
└── README.md
```
## ⚙️ Retrieval Pipeline
Query rewritten for clarity  
Hybrid retrieval generates candidates  
Results merged and deduplicated  
Neural reranker scores relevance  
Multi-hop expansion retrieves missing context  
Final context passed to LLM  

## 📊 Evaluation (Planned)

Designed for benchmarking with:  
Recall@K  
MRR (Mean Reciprocal Rank)  
Answer Faithfulness  
Context Precision  
Evaluation module intentionally separated to allow dataset-agnostic testing.  

## 🛠️ Tech Stack
Python 3.11  
FastAPI + Inngest for event-driven workflow  
Qdrant vector DB  
BM25 sparse search  
Cross-encoder reranker  
OpenAI embeddings & GPT-5 nano  
Streamlit UI for PDF upload and query  
Modular RAG architecture  

## ▶️ Quick Start
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/RAG.git
cd RAG

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -d --name qdrantRagDB -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# Run server (FastAPI + Inngest)
uvicorn main:app --reload

# Optional: Streamlit UI
streamlit run streamlit_app.py
```

## 📈 Engineering Highlights
Production-style retrieval orchestration  
Separation of retrieval vs reasoning layers  
Failure-resilient pipeline design  
Research-friendly experimentation structure  
Scalable to distributed vector databases  

## 🧩 Future Work
Retrieval evaluation dashboard  
Adaptive chunking  
Agentic retrieval planning  
Domain-specific fine-tuned reranker  

## 👤 Author
AI Engineer focused on information retrieval, document intelligence, and applied LLM systems.

## 📜 License
MIT License