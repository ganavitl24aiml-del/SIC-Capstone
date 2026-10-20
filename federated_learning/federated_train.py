
"""
Reusable FedAvg training script.

This reproduces the general experimental setup:
- 5 simulated clients
- ResNet18
- 6 disease labels
- BCEWithLogitsLoss
- Adam
- 1 local epoch
- sample-weighted FedAvg
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import (
    ChestXrayDataset,
    TARGET_DISEASES,
    IMAGE_DIR,
    transform
)

from model import create_model, get_device
from client import train_local_model
from server import fedavg


NUM_CLIENTS = 5
NUM_ROUNDS = 10
BATCH_SIZE = 32
LOCAL_EPOCHS = 1
LEARNING_RATE = 1e-4
RANDOM_SEED = 42


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "chest_xray"
)

SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    "federated_learning",
    "saved_models"
)

os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_data():

    train_df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "train_labeled.csv"
        )
    )

    val_df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "val_labeled.csv"
        )
    )

    return train_df, val_df


def create_clients(train_df):

    # Patient-level assignment
    patients = train_df["Patient ID"].unique()

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(patients)

    patient_groups = np.array_split(
        patients,
        NUM_CLIENTS
    )

    client_dfs = []

    for group in patient_groups:

        client_df = train_df[
            train_df["Patient ID"].isin(group)
        ].copy()

        client_dfs.append(client_df)

    return client_dfs


def evaluate_loss(model, loader, device):

    criterion = torch.nn.BCEWithLogitsLoss()

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


def run_federated_training():

    set_seed()

    device = get_device()

    print("Device:", device)

    train_df, val_df = load_training_data()

    client_dfs = create_clients(train_df)

    val_dataset = ChestXrayDataset(
        val_df,
        IMAGE_DIR,
        transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    global_model = create_model().to(device)

    history = []

    best_val_loss = float("inf")

    for round_num in range(
        1,
        NUM_ROUNDS + 1
    ):

        print(
            f"\n========== FedAvg Round "
            f"{round_num} =========="
        )

        local_states = []
        client_sizes = []
        client_losses = []

        for client_id, client_df in enumerate(
            client_dfs,
            start=1
        ):

            dataset = ChestXrayDataset(
                client_df,
                IMAGE_DIR,
                transform
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
            )

            local_model = create_model()

            local_model.load_state_dict(
                global_model.state_dict()
            )

            local_model, loss = train_local_model(
                local_model,
                loader,
                device,
                epochs=LOCAL_EPOCHS,
                learning_rate=LEARNING_RATE
            )

            local_states.append(
                local_model.state_dict()
            )

            client_sizes.append(
                len(client_df)
            )

            client_losses.append(loss)

            print(
                f"Client {client_id}: "
                f"Loss={loss:.4f}"
            )

        # FedAvg aggregation
        aggregated_state = fedavg(
            local_states,
            client_sizes
        )

        global_model.load_state_dict(
            aggregated_state
        )

        val_loss = evaluate_loss(
            global_model,
            val_loader,
            device
        )

        avg_client_loss = np.average(
            client_losses,
            weights=client_sizes
        )

        print(
            f"Average Client Loss: "
            f"{avg_client_loss:.4f}"
        )

        print(
            f"Global Validation Loss: "
            f"{val_loss:.4f}"
        )

        history.append({
            "round": round_num,
            "avg_client_loss": avg_client_loss,
            "validation_loss": val_loss
        })

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                global_model.state_dict(),
                os.path.join(
                    SAVE_DIR,
                    "fedavg_best_model_reproducible.pth"
                )
            )

            print("✓ Best model saved")

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        os.path.join(
            SAVE_DIR,
            "fedavg_reproducible_history.csv"
        ),
        index=False
    )

    print("\nTraining completed.")

    return global_model, history_df


if __name__ == "__main__":

    run_federated_training()
