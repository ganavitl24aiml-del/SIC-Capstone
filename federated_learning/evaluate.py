
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from dataset import (
    ChestXrayDataset,
    TARGET_DISEASES,
    IMAGE_DIR,
    transform
)

from model import create_model


def load_trained_model(
    model_path,
    device=None
):

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = create_model()

    state = torch.load(
        model_path,
        map_location=device
    )

    # Supports either a state_dict or a checkpoint
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)

    model = model.to(device)

    model.eval()

    return model


def get_predictions(
    model,
    data_loader,
    device
):

    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.sigmoid(
                outputs
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    probabilities = np.vstack(
        all_probabilities
    )

    labels = np.vstack(
        all_labels
    )

    return labels, probabilities


def calculate_metrics(
    y_true,
    probabilities,
    thresholds=None
):

    if thresholds is None:
        thresholds = np.full(
            len(TARGET_DISEASES),
            0.5
        )

    predictions = (
        probabilities >= thresholds
    ).astype(int)

    metrics = []

    for i, disease in enumerate(
        TARGET_DISEASES
    ):

        try:
            auc = roc_auc_score(
                y_true[:, i],
                probabilities[:, i]
            )
        except ValueError:
            auc = np.nan

        metrics.append({
            "Disease": disease,
            "F1": f1_score(
                y_true[:, i],
                predictions[:, i],
                zero_division=0
            ),
            "Precision": precision_score(
                y_true[:, i],
                predictions[:, i],
                zero_division=0
            ),
            "Recall": recall_score(
                y_true[:, i],
                predictions[:, i],
                zero_division=0
            ),
            "ROC_AUC": auc
        })

    metrics_df = pd.DataFrame(metrics)

    macro = {
        "Macro_F1": metrics_df["F1"].mean(),
        "Macro_Precision": metrics_df["Precision"].mean(),
        "Macro_Recall": metrics_df["Recall"].mean(),
        "Macro_ROC_AUC": metrics_df["ROC_AUC"].mean()
    }

    return metrics_df, macro


def find_best_thresholds(
    y_true,
    probabilities,
    threshold_values=None
):

    if threshold_values is None:

        threshold_values = np.arange(
            0.1,
            0.91,
            0.05
        )

    best_thresholds = []

    for i, disease in enumerate(
        TARGET_DISEASES
    ):

        best_threshold = 0.5
        best_f1 = -1

        for threshold in threshold_values:

            predictions = (
                probabilities[:, i]
                >= threshold
            ).astype(int)

            score = f1_score(
                y_true[:, i],
                predictions,
                zero_division=0
            )

            if score > best_f1:

                best_f1 = score
                best_threshold = threshold

        best_thresholds.append({
            "Disease": disease,
            "Threshold": best_threshold,
            "Validation_F1": best_f1
        })

    return pd.DataFrame(
        best_thresholds
    )


def evaluate_csv_model(
    model_path,
    csv_path,
    batch_size=32
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    df = pd.read_csv(csv_path)

    dataset = ChestXrayDataset(
        df,
        IMAGE_DIR,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = load_trained_model(
        model_path,
        device
    )

    y_true, probabilities = get_predictions(
        model,
        loader,
        device
    )

    metrics_df, macro = calculate_metrics(
        y_true,
        probabilities
    )

    return metrics_df, macro
