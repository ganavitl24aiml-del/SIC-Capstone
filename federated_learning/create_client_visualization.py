import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_CSV = PROJECT_ROOT / "data" / "chest_xray" / "train_labeled.csv"
SOURCE_IMAGES = PROJECT_ROOT / "data" / "chest_xray" / "images"

# IMPORTANT:
# This is a separate visualization copy.
# Your original images are NOT modified.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "chest_xray"
    / "client_visualization"
)

# ============================================================
# SETTINGS
# ============================================================

NUM_CLIENTS = 5
RANDOM_SEED = 42

DISEASES = [
    "Pneumonia",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Nodule",
    "Infiltration"
]

# Expected results from your existing FL split
EXPECTED = {
    1: {
        "Images": 965,
        "Patients": 611,
        "Pneumonia": 126,
        "Atelectasis": 239,
        "Cardiomegaly": 165,
        "Effusion": 298,
        "Nodule": 163,
        "Infiltration": 335
    },
    2: {
        "Images": 900,
        "Patients": 610,
        "Pneumonia": 136,
        "Atelectasis": 225,
        "Cardiomegaly": 153,
        "Effusion": 250,
        "Nodule": 167,
        "Infiltration": 335
    },
    3: {
        "Images": 947,
        "Patients": 610,
        "Pneumonia": 142,
        "Atelectasis": 252,
        "Cardiomegaly": 127,
        "Effusion": 310,
        "Nodule": 176,
        "Infiltration": 308
    },
    4: {
        "Images": 962,
        "Patients": 610,
        "Pneumonia": 145,
        "Atelectasis": 232,
        "Cardiomegaly": 195,
        "Effusion": 264,
        "Nodule": 150,
        "Infiltration": 303
    },
    5: {
        "Images": 947,
        "Patients": 610,
        "Pneumonia": 148,
        "Atelectasis": 227,
        "Cardiomegaly": 142,
        "Effusion": 296,
        "Nodule": 181,
        "Infiltration": 325
    }
}

# ============================================================
# CHECK SOURCE FILES
# ============================================================

print("\n========================================")
print("CLIENT VISUALIZATION SETUP")
print("========================================\n")

if not SOURCE_CSV.exists():
    raise FileNotFoundError(
        f"Training CSV not found:\n{SOURCE_CSV}"
    )

if not SOURCE_IMAGES.exists():
    raise FileNotFoundError(
        f"Image directory not found:\n{SOURCE_IMAGES}"
    )

print("Training CSV:")
print(SOURCE_CSV)

print("\nImage directory:")
print(SOURCE_IMAGES)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(SOURCE_CSV)

print(f"\nTotal training images: {len(df):,}")

required_columns = [
    "Image Index",
    "Patient ID"
] + DISEASES

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in train_labeled.csv: {missing}"
    )

# ============================================================
# RECREATE ORIGINAL CLIENT ASSIGNMENT
# ============================================================

print("\nRecreating existing patient-level client split...")
print(f"Random seed: {RANDOM_SEED}")

patients = df["Patient ID"].astype(str).unique()

print(f"Unique patients: {len(patients):,}")

# This reproduces the original style of split:
# shuffle patients using seed 42, then divide into 5 clients.

np.random.seed(RANDOM_SEED)
np.random.shuffle(patients)

client_patients = np.array_split(
    patients,
    NUM_CLIENTS
)

# ============================================================
# CREATE ASSIGNMENTS
# ============================================================

df = df.copy()

df["Patient ID"] = df["Patient ID"].astype(str)

patient_to_client = {}

for client_number, patient_list in enumerate(
    client_patients,
    start=1
):
    for patient_id in patient_list:
        patient_to_client[patient_id] = client_number

df["Client"] = df["Patient ID"].map(patient_to_client)

if df["Client"].isna().any():
    raise RuntimeError(
        "Some images were not assigned to a client."
    )

# ============================================================
# VALIDATE EXISTING DISTRIBUTION
# ============================================================

print("\n========================================")
print("VALIDATING EXISTING CLIENT SPLIT")
print("========================================\n")

all_correct = True

for client_number in range(1, NUM_CLIENTS + 1):

    client_df = df[
        df["Client"] == client_number
    ].copy()

    actual = {
        "Images": len(client_df),
        "Patients": client_df["Patient ID"].nunique()
    }

    for disease in DISEASES:
        actual[disease] = int(
            client_df[disease].sum()
        )

    expected = EXPECTED[client_number]

    print(f"Client {client_number}")

    for key in expected:

        status = "OK"

        if actual[key] != expected[key]:
            status = "MISMATCH"
            all_correct = False

        print(
            f"  {key:15s} "
            f"Expected={expected[key]:4d} "
            f"Actual={actual[key]:4d} "
            f"[{status}]"
        )

    print()

# ============================================================
# SAFETY CHECK
# ============================================================

if not all_correct:

    print("========================================")
    print("WARNING")
    print("========================================")

    print(
        "\nThe recreated assignment does NOT exactly match "
        "the client distribution you previously recorded."
    )

    print(
        "\nNothing has been copied."
    )

    print(
        "\nThis means your original client split used a "
        "slightly different assignment procedure."
    )

    print(
        "\nDO NOT continue by changing the data."
    )

    print(
        "\nSend me the output above and we will reproduce "
        "the exact original split."
    )

    raise SystemExit

# ============================================================
# REMOVE ONLY OLD VISUALIZATION COPY
# ============================================================

if OUTPUT_DIR.exists():

    print(
        f"\nRemoving previous visualization copy:\n"
        f"{OUTPUT_DIR}"
    )

    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# CREATE CLIENT FOLDERS
# ============================================================

print("\n========================================")
print("CREATING CLIENT VISUALIZATION DATA")
print("========================================\n")

for client_number in range(1, NUM_CLIENTS + 1):

    client_df = df[
        df["Client"] == client_number
    ].copy()

    client_dir = (
        OUTPUT_DIR
        / f"client_{client_number}"
    )

    image_dir = client_dir / "images"

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save labels
    client_df.to_csv(
        client_dir / "labels.csv",
        index=False
    )

    print(
        f"Client {client_number}: "
        f"{len(client_df):,} images"
    )

    # ========================================================
    # COPY IMAGES
    # ========================================================

    missing_images = []

    for count, (_, row) in enumerate(
        client_df.iterrows(),
        start=1
    ):

        image_name = row["Image Index"]

        source = SOURCE_IMAGES / image_name
        destination = image_dir / image_name

        if not source.exists():

            missing_images.append(image_name)
            continue

        shutil.copy2(
            source,
            destination
        )

        if count % 100 == 0:
            print(
                f"   copied {count:,} / "
                f"{len(client_df):,}"
            )

    # Save missing image list if needed
    if missing_images:

        pd.DataFrame({
            "Image Index": missing_images
        }).to_csv(
            client_dir / "missing_images.csv",
            index=False
        )

        print(
            f"   WARNING: "
            f"{len(missing_images)} images missing"
        )

    else:

        print("   All images copied successfully.")

# ============================================================
# CREATE CLIENT SUMMARY
# ============================================================

summary_rows = []

for client_number in range(1, NUM_CLIENTS + 1):

    client_df = df[
        df["Client"] == client_number
    ]

    row = {
        "Client": f"Client {client_number}",
        "Images": len(client_df),
        "Patients": client_df["Patient ID"].nunique()
    }

    for disease in DISEASES:
        row[disease] = int(
            client_df[disease].sum()
        )

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "client_distribution.csv",
    index=False
)

# ============================================================
# SAVE COMPLETE ASSIGNMENT
# ============================================================

df[
    [
        "Image Index",
        "Patient ID",
        "Client"
    ] + DISEASES
].to_csv(
    OUTPUT_DIR / "image_client_assignments.csv",
    index=False
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========================================")
print("SUCCESS")
print("========================================")

print(
    f"\nVisualization data created at:\n"
    f"{OUTPUT_DIR}"
)

print("\nStructure:")

for client_number in range(1, NUM_CLIENTS + 1):

    print(
        f"  client_{client_number}/"
    )
    print(
        f"      ├── images/"
    )
    print(
        f"      └── labels.csv"
    )

print(
    "\nAdditional files:"
)

print(
    "  client_distribution.csv"
)

print(
    "  image_client_assignments.csv"
)

print(
    "\nYour original dataset and FedAvg results "
    "were not modified."
)