import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from rag_utils import rag_pipeline

st.set_page_config(page_title="Research Paper RAG", layout="wide")

st.title("📄 Research Paper RAG Assistant")
st.write("Ask questions over arXiv research papers")

@st.cache_resource
def load_assets():
    index = faiss.read_index("faiss_index.bin")
    with open("data_bundle.pkl", "rb") as f:
        data = pickle.load(f)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return index, data, model

index, data, embedding_model = load_assets()

query = st.text_input("Enter your question")

if query:
    context, sources, confidence = rag_pipeline(
        query,
        embedding_model,
        index,
        data["documents"],
        data["titles"],
        data["categories"],
        top_k=1
    )

    st.subheader("Confidence")
    st.write(f"{confidence}%")

    st.subheader("Sources")
    for s in sources:
        st.write(f"**Title:** {s['title']}")
        st.write(f"**Category:** {s['categories']}")
        st.write(f"**Similarity:** {round(s['similarity'], 3)}")
        st.markdown("---")

    st.subheader("Context Used")
    st.text_area("Retrieved Context", context, height=300)
