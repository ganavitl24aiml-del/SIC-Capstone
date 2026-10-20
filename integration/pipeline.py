import os
import sys

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT EXISTING COMPONENTS
# ============================================================

from integration.prediction import (
    load_fedavg_model,
    predict_xray,
    load_tuned_thresholds
)

from integration.rag_pipeline import (
    ask_medical_rag
)


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("Loading Federated Learning model...")

MODEL, DEVICE = load_fedavg_model()

THRESHOLDS = load_tuned_thresholds()

print("✓ Federated Learning model loaded")


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_xray(
    image_path,
    rag_top_k=5
):
    """
    Complete clinical decision-support pipeline.

    1. Predict diseases using the trained FedAvg model.
    2. Identify positive predictions.
    3. Retrieve medical evidence using RAG.
    4. Generate a patient-friendly explanation.
    """

    # --------------------------------------------------------
    # STEP 1: FEDERATED MODEL PREDICTION
    # --------------------------------------------------------

    predictions = predict_xray(
        image_path=image_path,
        model=MODEL,
        device=DEVICE,
        thresholds=THRESHOLDS
    )


    # --------------------------------------------------------
    # STEP 2: IDENTIFY PREDICTED DISEASES
    # --------------------------------------------------------

    predicted_diseases = [
        result["disease"]
        for result in predictions
        if result["predicted"]
    ]


    # --------------------------------------------------------
    # STEP 3: CREATE RAG QUESTION
    # --------------------------------------------------------

    if predicted_diseases:

        disease_text = ", ".join(
            predicted_diseases
        )

        question = (
    f"What chest X-ray findings and general clinical "
    f"features are associated with {disease_text}? "
    f"Focus on information relevant to chest X-ray "
    f"clinical decision support."
)

    else:
        question = (
        "The chest X-ray model did not predict any of "
        "the target diseases. Based only on the provided "
        "medical sources, explain what information is "
        "available about the possible chest X-ray findings. "
        "Do not assume that any disease is present."
    )


    # --------------------------------------------------------
    # STEP 4: RETRIEVE + GENERATE
    # --------------------------------------------------------

    rag_result = ask_medical_rag(
        question,
        k=rag_top_k
    )


    # --------------------------------------------------------
    # STEP 5: RETURN COMPLETE RESULT
    # --------------------------------------------------------

    return {
        "image_path": image_path,
        "predictions": predictions,
        "predicted_diseases": predicted_diseases,
        "rag_question": question,
        "answer": rag_result["answer"],
        "sources": rag_result["sources"]
    }


# ============================================================
# TEST PIPELINE
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("FEDERATED + RAG PIPELINE TEST")
    print("========================================")

    image_path = input(
        "\nEnter path to a chest X-ray image: "
    ).strip().strip('"')

    if not os.path.exists(image_path):

        print(
            "\nERROR: Image not found."
        )

        print(
            "Please provide a valid image path."
        )

        sys.exit(1)


    print("\nAnalyzing X-ray...")
    print("----------------------------------------")

    result = analyze_xray(
        image_path
    )


    # --------------------------------------------------------
    # DISPLAY PREDICTIONS
    # --------------------------------------------------------

    print("\nDisease Predictions:")
    print("----------------------------------------")

    for prediction in result["predictions"]:

        status = (
            "PREDICTED"
            if prediction["predicted"]
            else "Not predicted"
        )

        print(
            f"{prediction['disease']}: "
            f"{prediction['probability']:.3f} "
            f"→ {status}"
        )


    # --------------------------------------------------------
    # DISPLAY RAG ANSWER
    # --------------------------------------------------------

    print("\nPredicted Diseases:")
    print("----------------------------------------")

    if result["predicted_diseases"]:

        for disease in result["predicted_diseases"]:
            print("•", disease)

    else:

        print("No target disease crossed the threshold.")


    print("\nRAG Explanation:")
    print("----------------------------------------")

    print(
        result["answer"]
    )


    # --------------------------------------------------------
    # DISPLAY SOURCES
    # --------------------------------------------------------

    print("\nMedical Sources:")
    print("----------------------------------------")

    for source in result["sources"]:

        print(
            "•",
            source["source"]
        )

    print(
        "\n========================================"
    )
    print("PIPELINE TEST COMPLETE")
    print("========================================")