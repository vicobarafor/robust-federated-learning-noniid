from __future__ import annotations

from typing import List, Dict
from copy import deepcopy

import torch
import torch.nn as nn


def average_weights(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Standard FedAvg weight averaging."""
    avg_state = deepcopy(state_dicts[0])
    for k in avg_state.keys():
        for i in range(1, len(state_dicts)):
            avg_state[k] += state_dicts[i][k]
        avg_state[k] = avg_state[k] / len(state_dicts)
    return avg_state


def local_train(
    model: nn.Module,
    loader,
    device: str,
    epochs: int,
    lr: float,
    mu: float = 0.0,
    global_model: nn.Module | None = None,
) -> Dict[str, torch.Tensor]:
    """
    FedAvg local training if mu=0.
    FedProx local training if mu>0: adds (mu/2)*||w - w_global||^2.
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    global_params = None
    if mu > 0.0:
        if global_model is None:
            raise ValueError("FedProx requires global_model when mu > 0.")
        global_model = global_model.to(device)
        global_model.eval()
        global_params = [p.detach().clone() for p in global_model.parameters()]

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)

            if mu > 0.0 and global_params is not None:
                prox = 0.0
                for p, gp in zip(model.parameters(), global_params):
                    prox = prox + torch.sum((p - gp) ** 2)
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()

    return model.state_dict()