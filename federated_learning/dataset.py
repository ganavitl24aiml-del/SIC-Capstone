
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# Six diseases used in the FL model
TARGET_DISEASES = [
    "Pneumonia",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Nodule",
    "Infiltration"
]


# Project-relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "chest_xray"
)

IMAGE_DIR = os.path.join(DATA_DIR, "images")


# Same preprocessing used during FL training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ChestXrayDataset(Dataset):

    def __init__(self, dataframe, image_dir=IMAGE_DIR, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = os.path.join(
            self.image_dir,
            row["Image Index"]
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[TARGET_DISEASES].values.astype(float),
            dtype=torch.float32
        )

        return image, labels


def load_split(split):

    csv_path = os.path.join(
        DATA_DIR,
        f"{split}_labeled.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}"
        )

    return pd.read_csv(csv_path)
