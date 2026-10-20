import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

from rag.retrieval import retrieve
from rag.generation import generate_answer


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

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


# --------------------------------------------------
# Load existing vector database
# --------------------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(INDEX_PATH)

print("Loading chunks...")

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)


# --------------------------------------------------
# Load same embedding model used to build the database
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Test retrieval
# --------------------------------------------------

question = "What are common clinical features of pneumonia?"

print("\nQuestion:")
print(question)

results = retrieve(
    question,
    index,
    chunks,
    embedding_model,
    k=5
)


print("\nRetrieved results:")

for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(result)


# --------------------------------------------------
# Test LLM generation
# --------------------------------------------------

print("\nGenerating answer...")

formatted_results = []

for result in results:
    formatted_results.append({
        "source": result["filename"],
        "text": result["text"]
    })

answer = generate_answer(
    question,
    formatted_results
)

print("\nAnswer:")
print(answer)