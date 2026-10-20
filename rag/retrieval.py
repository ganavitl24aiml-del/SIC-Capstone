import numpy as np


def retrieve(
    question,
    index,
    chunks,
    embedding_model,
    k=5,
    search_k=20
):
    """
    Retrieve relevant medical chunks while avoiding
    repeated source documents.

    k = number of final results
    search_k = number of FAISS candidates considered
    """

    # Create embedding for the question
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    # Search for more candidates than we finally need
    distances, indices = index.search(
        query_embedding,
        search_k
    )

    results = []

    # Keep track of documents already selected
    used_sources = set()

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        # Ignore invalid FAISS indices
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        # Get source information
        source = chunk.get(
            "filename",
            "Unknown source"
        )

        disease = chunk.get(
            "disease",
            "Unknown"
        )

        text = chunk.get(
            "text",
            ""
        )

        # Skip empty chunks
        if not text.strip():
            continue

        # Avoid duplicate source documents
        if source in used_sources:
            continue

        used_sources.add(source)

        results.append({
            "disease": disease,
            "filename": source,
            "text": text,
            "distance": float(distance)
        })

        # Stop once we have enough unique sources
        if len(results) >= k:
            break

    return results