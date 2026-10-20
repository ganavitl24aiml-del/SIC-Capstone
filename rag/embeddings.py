
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():

    return SentenceTransformer(
        MODEL_NAME
    )


def create_embeddings(
    chunks,
    model=None
):

    if model is None:
        model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )
