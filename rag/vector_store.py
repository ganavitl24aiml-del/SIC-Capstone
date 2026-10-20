
import os
import pickle
import faiss


def create_vector_index(
    embeddings
):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


def save_vector_store(
    index,
    chunks,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    index_path = os.path.join(
        output_dir,
        "medical_index.faiss"
    )

    chunks_path = os.path.join(
        output_dir,
        "chunks.pkl"
    )

    faiss.write_index(
        index,
        index_path
    )

    with open(
        chunks_path,
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )

    print(
        "Vector index saved:",
        index_path
    )

    print(
        "Chunks saved:",
        chunks_path
    )


def load_vector_store(
    vector_dir
):

    index_path = os.path.join(
        vector_dir,
        "medical_index.faiss"
    )

    chunks_path = os.path.join(
        vector_dir,
        "chunks.pkl"
    )

    index = faiss.read_index(
        index_path
    )

    with open(
        chunks_path,
        "rb"
    ) as f:

        chunks = pickle.load(f)

    return index, chunks
