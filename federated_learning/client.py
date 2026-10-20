
import torch
import torch.nn as nn
import torch.optim as optim


def train_local_model(
    model,
    train_loader,
    device,
    epochs=1,
    learning_rate=1e-4
):

    model = model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    total_loss = 0.0

    for epoch in range(epochs):

        epoch_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        total_loss = epoch_loss

    return model, total_loss


def get_model_parameters(model):

    return [
        parameter.detach().cpu().numpy()
        for parameter in model.state_dict().values()
    ]


def set_model_parameters(model, parameters):

    state_dict = model.state_dict()

    for key, value in zip(
        state_dict.keys(),
        parameters
    ):

        state_dict[key] = torch.tensor(value)

    model.load_state_dict(state_dict)

    return model
