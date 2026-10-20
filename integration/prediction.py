
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


# ------------------------------------------------------------
# Make federated_learning modules importable
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FL_DIR = os.path.join(
    PROJECT_ROOT,
    "federated_learning"
)

if FL_DIR not in sys.path:
    sys.path.insert(0, FL_DIR)


from model import create_model
from dataset import TARGET_DISEASES


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "federated_learning",
    "saved_models",
    "fedavg_best_model.pth"
)


# ------------------------------------------------------------
# Same preprocessing used during training
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# Load the already-trained FedAvg model
# ------------------------------------------------------------

def load_fedavg_model(
    model_path=MODEL_PATH,
    device=None
):

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Trained model not found:\n{model_path}"
        )

    model = create_model()

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # Handle a normal state_dict
    # Handle saved checkpoint
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            print("Loading saved FedAvg checkpoint...")
            print("Best round:", checkpoint.get("best_round"))
            print("Best validation loss:", checkpoint.get("best_val_loss"))

            checkpoint = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, device


# ------------------------------------------------------------
# Predict from an X-ray image
# ------------------------------------------------------------

def predict_xray(
    image_path,
    model=None,
    device=None,
    thresholds=None
):

    if model is None:

        model, device = load_fedavg_model()

    if device is None:

        device = next(
            model.parameters()
        ).device

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()[0]

    # Default threshold
    if thresholds is None:

        thresholds = np.full(
            len(TARGET_DISEASES),
            0.5
        )

    predictions = (
        probabilities >= thresholds
    )

    results = []

    for disease, probability, prediction in zip(
        TARGET_DISEASES,
        probabilities,
        predictions
    ):

        results.append({
            "disease": disease,
            "probability": float(probability),
            "predicted": bool(prediction)
        })

    return results


# ------------------------------------------------------------
# Load your experimentally tuned thresholds
# ------------------------------------------------------------

def load_tuned_thresholds(
    threshold_file=None
):
    if threshold_file is None:
        threshold_file = os.path.join(
            PROJECT_ROOT,
            "federated_learning",
            "saved_models",
            "fedavg_thresholds.csv"
        )

    if not os.path.exists(threshold_file):
        print(
            "Threshold file not found. "
            "Using default threshold 0.5."
        )

        return np.full(
            len(TARGET_DISEASES),
            0.5,
            dtype=np.float32
        )

    import pandas as pd

    # The threshold CSV has NO header.
    df = pd.read_csv(
        threshold_file,
        header=None,
        names=["Disease", "Threshold"]
    )

    # Convert threshold values to numbers
    df["Threshold"] = pd.to_numeric(
        df["Threshold"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=["Threshold"]
    )

    threshold_map = dict(
        zip(
            df["Disease"].astype(str).str.strip(),
            df["Threshold"]
        )
    )

    thresholds = np.array([
        threshold_map.get(
            disease,
            0.5
        )
        for disease in TARGET_DISEASES
    ], dtype=np.float32)

    print("Loaded thresholds:")
    for disease, threshold in zip(
        TARGET_DISEASES,
        thresholds
    ):
        print(
            f"  {disease}: {threshold:.2f}"
        )

    return thresholds

# ------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------

def predict_with_tuned_thresholds(
    image_path
):

    model, device = load_fedavg_model()

    thresholds = load_tuned_thresholds()

    return predict_xray(
        image_path=image_path,
        model=model,
        device=device,
        thresholds=thresholds
    )


# ------------------------------------------------------------
# Simple test
# ------------------------------------------------------------

if __name__ == "__main__":

    print("FedAvg prediction module loaded.")
    print("Model:", MODEL_PATH)
    print("Diseases:")

    for disease in TARGET_DISEASES:
        print(" -", disease)
