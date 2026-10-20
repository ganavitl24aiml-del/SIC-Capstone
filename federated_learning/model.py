
import torch
import torch.nn as nn
from torchvision import models


NUM_CLASSES = 6


def create_model():

    model = models.resnet18(weights="DEFAULT")

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


class DPCNN(nn.Module):
    """
    Lightweight CNN used for the Differential Privacy
    proof-of-concept with Opacus.
    """

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        return self.classifier(x)


def get_device():

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
