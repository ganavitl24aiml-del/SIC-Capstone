import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2"


def build_prompt(question, retrieved_results):

    context_parts = []

    for i, result in enumerate(
        retrieved_results,
        start=1
    ):
        context_parts.append(
            f"[Source {i}: {result['source']}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a medical clinical decision-support assistant.

Your job is to explain information from the provided
medical sources in a clear, simple, patient-friendly way.

IMPORTANT RULES:

1. Use ONLY the information contained in the provided
   medical sources.

2. Do NOT invent, assume, or add medical information
   from your general knowledge.

3. The chest X-ray model prediction is NOT a diagnosis.
   Never present a predicted disease as confirmed.

4. Focus on information relevant to:
   - chest X-ray findings
   - relevant clinical features
   - symptoms or signs mentioned in the sources
   - general medical context explicitly supported by
     the sources

5. Avoid unnecessary discussion of other imaging
   modalities such as:
   - ultrasound
   - CT
   - MRI
   - PET
   unless the retrieved sources specifically contain
   information that is directly necessary to answer
   the question.

6. Do NOT provide treatment recommendations unless
   the provided sources explicitly contain relevant
   treatment information and the question specifically
   asks about treatment.

7. Do NOT give a diagnosis, prognosis, or personalized
   medical advice.

8. If the retrieved sources do not contain enough
   information to answer the question, say exactly:

"Information not found in the provided medical sources."

9. Do not mention the retrieval process, embeddings,
   FAISS, vector databases, or the language model.

10. Do not repeat the question.

11. Keep the answer concise and easy to understand.

12. When several sources provide the same information,
    combine them instead of repeating it.

13. If sources contain conflicting information, clearly
    state that the provided sources differ instead of
    choosing an unsupported answer.

14. Use cautious language such as:
    "may be associated with"
    "can be seen with"
    "the provided sources describe"
    rather than saying something is definitely present.

ANSWER FORMAT:

Use this structure when the information is available:

Summary:
Give a short 1–2 sentence explanation.

Key points:
- Give the most relevant points from the sources.
- Keep each point short and understandable.
- Include chest X-ray findings when the sources provide them.

Important:
End with a short reminder that the information is
educational/decision-support information and is not
a medical diagnosis.

Question:
{question}

Medical sources:
{context}

Answer:
"""

    return prompt


def generate_answer(
    question,
    retrieved_results,
    model_name=MODEL_NAME
):

    if not retrieved_results:
        return (
            "Information not found in the provided "
            "medical sources."
        )

    prompt = build_prompt(
        question,
        retrieved_results
    )

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9
                }
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "response",
            ""
        ).strip()

        if not answer:
            return (
                "Information not found in the provided "
                "medical sources."
            )

        return answer

    except requests.exceptions.ConnectionError:

        return (
            "Unable to connect to the local medical "
            "language model. Please make sure Ollama "
            "is running."
        )

    except requests.exceptions.Timeout:

        return (
            "The medical explanation took too long to "
            "generate. Please try again."
        )

    except requests.exceptions.RequestException as e:

        return (
            f"Unable to generate the medical explanation: "
            f"{str(e)}"
        )


def rag_answer(
    question,
    index,
    chunks,
    embedding_model,
    k=5
):

    from retrieval import retrieve

    results = retrieve(
        question,
        index,
        chunks,
        embedding_model,
        k=k
    )

    answer = generate_answer(
        question,
        results
    )

    return answer, results


if __name__ == "__main__":

    print(
        "Medical RAG generation module loaded."
    )