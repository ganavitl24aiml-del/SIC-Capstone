
import torch


def fedavg(state_dicts, client_sizes):
    """
    Sample-weighted Federated Averaging.

    state_dicts:
        List of model state dictionaries from clients.

    client_sizes:
        Number of training samples used by each client.
    """

    if len(state_dicts) != len(client_sizes):
        raise ValueError(
            "Number of models and client sizes must match."
        )

    total_samples = sum(client_sizes)

    global_state = {}

    for key in state_dicts[0].keys():

        weighted_parameter = torch.zeros_like(
            state_dicts[0][key],
            dtype=torch.float32
        )

        for state_dict, size in zip(
            state_dicts,
            client_sizes
        ):

            weighted_parameter += (
                state_dict[key].float()
                * (size / total_samples)
            )

        global_state[key] = weighted_parameter

    return global_state


def aggregate_models(global_model, local_models, client_sizes):

    state_dicts = [
        model.state_dict()
        for model in local_models
    ]

    aggregated_state = fedavg(
        state_dicts,
        client_sizes
    )

    global_model.load_state_dict(
        aggregated_state
    )

    return global_model
