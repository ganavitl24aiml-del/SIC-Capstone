
"""
Differential Privacy proof-of-concept.

The DP experiment used a separate lightweight CNN because
the original ResNet18 architecture was not compatible with
the Opacus validation used during the experiment.

Important:
This file represents the DP prototype/experiment and should
not be interpreted as a rigorous end-to-end privacy guarantee
across all federated rounds.
"""

import torch
import torch.nn as nn


class DPCNN(nn.Module):

    def __init__(self, num_classes=6):

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


def create_dp_model():

    return DPCNN(num_classes=6)


def attach_privacy_engine(
    model,
    optimizer,
    data_loader,
    target_epsilon=8.0,
    target_delta=1e-5,
    max_grad_norm=1.0
):

    from opacus import PrivacyEngine

    privacy_engine = PrivacyEngine()

    model, optimizer, data_loader = (
        privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm
        )
    )

    return (
        model,
        optimizer,
        data_loader,
        privacy_engine
    )


def get_epsilon(
    privacy_engine,
    delta=1e-5
):

    return privacy_engine.get_epsilon(
        delta
    )
