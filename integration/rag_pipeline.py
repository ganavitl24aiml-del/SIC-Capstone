import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer

import os
import sys
import pickle
import faiss

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sentence_transformers import SentenceTransformer

from rag.retrieval import retrieve
from rag.generation import generate_answer



# ============================================================
# PROJECT PATHS
# ============================================================

INDEX_PATH = os.path.join(
    PROJECT_ROOT,
    "rag",
    "vector_db",
    "medical_index.faiss"
)

CHUNKS_PATH = os.path.join(
    PROJECT_ROOT,
    "rag",
    "vector_db",
    "chunks.pkl"
)


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

print("Loading RAG components...")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(
        f"FAISS index not found:\n{INDEX_PATH}"
    )

if not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(
        f"Chunks file not found:\n{CHUNKS_PATH}"
    )


index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("✓ RAG components loaded")


# ============================================================
# RAG FUNCTION
# ============================================================

def ask_medical_rag(
    question,
    k=5
):
    """
    Retrieve relevant medical evidence and generate
    a patient-friendly answer using Ollama.
    """

    # Retrieve relevant chunks
    results = retrieve(
        question,
        index,
        chunks,
        embedding_model,
        k=k
    )

    # Convert retrieval format to generation format
    formatted_results = []

    for result in results:

        formatted_results.append({
            "source": result["filename"],
            "text": result["text"]
        })

    # Generate answer
    answer = generate_answer(
        question,
        formatted_results
    )

    return {
        "question": question,
        "answer": answer,
        "sources": formatted_results
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "What are common clinical features of pneumonia?"
    )

    result = ask_medical_rag(question)

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    for source in result["sources"]:
        print("-", source["source"])