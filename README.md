# 📄 Research Paper RAG Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** system built on real **arXiv research papers**, designed to answer user questions with **source attribution** and **confidence scoring**.

This project demonstrates how modern GenAI systems are built in practice by combining:
- Transformer-based embeddings
- Vector databases (FAISS)
- Retrieval-first reasoning
- Honest uncertainty estimation
- Lightweight deployment using Streamlit

---

##  Live Demo
🔗 https://rag-research-assistant-pre-added.streamlit.app/

---

##  Problem Statement

Large Language Models (LLMs) often **hallucinate** when answering questions, especially in technical domains like research papers.

This project solves that problem by:
- Retrieving **relevant research documents first**
- Generating answers **only from retrieved context**
- Showing **which paper** was used
- Reporting a **confidence score** for transparency

---

##  What This System Does

1. Accepts a **natural language question**
2. Finds the most relevant research papers using **semantic search**
3. Builds a context from retrieved documents
4. Generates an answer **grounded in real papers**
5. Displays:
   - Confidence score
   - Source paper title & category
   - Context used for answering

---

##  System Architecture

### High-Level Architecture

User Query
↓
Sentence Transformer (Embeddings)
↓
FAISS Vector Index
↓
Top-K Relevant Documents
↓
Context Construction
↓
Confidence Scoring + Source Attribution
↓
Streamlit UI Output



---

##  Detailed RAG Pipeline Flow

```mermaid
flowchart TD
    A[User Query] --> B[Query Embedding]
    B --> C[FAISS Similarity Search]
    C --> D[Retrieve Top-K Documents]
    D --> E[Context Truncation]
    E --> F[Confidence Computation]
    F --> G[Display Answer + Sources]
💡 This flow ensures the model never answers without first retrieving evidence.

 Dataset

Source: arXiv (Cornell University)

Type: JSON Lines (.json)

Fields Used:

title

abstract

categories

Only abstracts are used to:

Reduce noise

Fit model context windows

Improve retrieval precision

 Embedding Model

Model: all-MiniLM-L6-v2

Why this model?

Lightweight

Strong semantic performance

Industry-standard for RAG systems

Computation:

GPU (Tesla T4) used during embedding generation

CPU used during inference & retrieval

 Vector Database (FAISS)

Index Type: IndexFlatIP

Similarity Metric: Cosine similarity (via normalized embeddings)

Why FAISS?

Extremely fast vector search

Widely used in production RAG systems

Scales to millions of documents

 Confidence Scoring

Confidence is derived from cosine similarity scores returned by FAISS.

confidence (%) = average_similarity × 100


This allows the system to:

Communicate uncertainty honestly

Avoid overconfident hallucinations

Build user trust

 Source Attribution

For every answer, the system displays:

Research paper title

arXiv category (e.g. hep-ph)

Similarity score

This makes the system transparent and auditable.

 User Interface (Streamlit)

The Streamlit app provides:

Text input for questions

Confidence percentage

Source paper details

Context preview used for answering

Designed to be:

Lightweight

Fast

Easy to deploy on Streamlit Cloud

 Project Structure
Research_Paper_RAG_Assistant/
│
├── app.py                 # Streamlit application
├── rag_utils.py           # Core RAG logic
├── faiss_index.bin        # Vector database
├── data_bundle.pkl        # Documents + metadata
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

 Example Questions to Try

What is diphoton production in particle physics?

What are next-to-leading order contributions?

How are photon pairs produced in hadron colliders?

What kind of problems are studied in hep-ph research?

Out-of-scope questions correctly return low confidence or "I don't know".

 Known Limitations

Uses abstracts only, not full PDFs

Answer generation is conservative by design

No fine-tuned LLM (uses retrieval-first approach)

These are intentional design choices for reliability.

 Future Improvements

Full PDF ingestion & chunking

Better answer generation models

Reranking retrieved documents

Multi-document synthesis

Evaluation dashboard (Precision@K)

 Skills Demonstrated

Retrieval-Augmented Generation (RAG)

Vector databases (FAISS)

Transformer embeddings

GPU/CPU workload separation

Explainable AI

Production-ready deployment

📜 License

This project is for educational and portfolio purposes.

 Acknowledgements

arXiv & Cornell University

Hugging Face

Facebook AI (FAISS)

Streamlit

 If you find this project useful, feel free to star the repository!


---

##  About Images & Diagrams (Important)

Right now:
- Mermaid diagram renders automatically on GitHub ✔
- Streamlit doesn’t need images ✔

Later (optional):
- Add `/assets/architecture.png`
- Add `/assets/rag_flow.png`
- Reference them in README like:

```markdown
![Architecture](assets/architecture.png)

