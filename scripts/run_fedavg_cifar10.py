from __future__ import annotations

import os
import json
import argparse
from copy import deepcopy
from typing import Dict, Any, List, Tuple

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.resnet_cifar import ResNetCIFAR
from src.fl.partition import make_client_loaders
from src.fl.fedavg import local_train, average_weights


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval().to(device)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def _state_dict_to_vector(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Flatten a state_dict into a single 1D vector (CPU float32).
    Only includes floating tensors (weights/biases), skips buffers like num_batches_tracked.
    """
    vecs: List[torch.Tensor] = []
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        if not v.is_floating_point():
            continue
        vecs.append(v.detach().to("cpu", dtype=torch.float32).flatten())
    if not vecs:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(vecs, dim=0)


def _model_num_bytes(model: torch.nn.Module) -> int:
    """
    Approximate model payload size in bytes (float params + buffers).
    This is what gets transmitted each round in a simple FL simulation.
    """
    total = 0
    with torch.no_grad():
        for p in model.parameters():
            total += p.numel() * p.element_size()
        for b in model.buffers():
            total += b.numel() * b.element_size()
    return int(total)


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- dataset ----
    dcfg = cfg["dataset"]
    batch_size = int(dcfg.get("batch_size", 64))
    eval_batch_size = int(dcfg.get("eval_batch_size", 256))

    train_loader, test_loader = get_cifar10_dataloaders(batch_size, eval_batch_size)
    train_ds = train_loader.dataset

    # ---- federated ----
    fcfg = cfg["federated"]
    num_clients = int(fcfg["num_clients"])
    clients_per_round = int(fcfg["clients_per_round"])
    rounds = int(fcfg["rounds"])
    local_epochs = int(fcfg["local_epochs"])
    lr = float(fcfg["lr"])
    mu = float(fcfg.get("mu", 0.0))  # 0 => FedAvg, >0 => FedProx

    # ---- partition ----
    pcfg = cfg.get("partition", {"type": "iid"})

    client_loaders = make_client_loaders(
        dataset=train_ds,
        num_clients=num_clients,
        batch_size=batch_size,
        partition_cfg=pcfg,
        seed=seed,
    )

    # ---- logging ----
    out_dir = cfg.get("logging", {}).get("out_dir", "results/cifar10_run")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # ---- model ----
    global_model = ResNetCIFAR(num_classes=10).to(device)
    model_bytes = _model_num_bytes(global_model)
    model_mb = model_bytes / (1024.0 * 1024.0)

    history: Dict[str, List[float]] = {
        "round": [],
        "test_acc": [],
        # communication
        "comm_round_mb": [],
        "comm_cum_mb": [],
        # drift diagnostics
        "drift_mean_l2": [],
        "drift_std_l2": [],
        "cosine_mean": [],
        "cosine_std": [],
        # helpful metadata
        "model_mb": [float(model_mb)],
    }

    rng = np.random.default_rng(seed)
    comm_cum_mb = 0.0

    for r in range(1, rounds + 1):
        # sample clients
        chosen = rng.choice(num_clients, size=clients_per_round, replace=False)

        global_state = deepcopy(global_model.state_dict())  # snapshot for deltas
        global_vec = _state_dict_to_vector(global_state)    # flatten once

        local_states: List[Dict[str, torch.Tensor]] = []
        deltas: List[torch.Tensor] = []

        for cid in chosen:
            local_model = ResNetCIFAR(num_classes=10)
            local_model.load_state_dict(global_state)

            sd = local_train(
                model=local_model,
                loader=client_loaders[cid],
                device=device,
                epochs=local_epochs,
                lr=lr,
                mu=mu,
                global_model=global_model if mu > 0 else None,
            )
            local_states.append(sd)

            local_vec = _state_dict_to_vector(sd)
            delta = local_vec - global_vec
            deltas.append(delta)

        # ---- aggregation ----
        new_state = average_weights(local_states)
        global_model.load_state_dict(new_state)

        # ---- evaluation ----
        acc = evaluate(global_model, test_loader, device=device)

        # ---- communication (simple FL accounting) ----
        # Each selected client downloads global model + uploads local model/update
        comm_round_mb = float(2.0 * clients_per_round * model_mb)
        comm_cum_mb += comm_round_mb

        # ---- drift diagnostics ----
        # mean delta
        if len(deltas) > 0:
            D = torch.stack(deltas, dim=0)  # [K, P]
            mean_delta = D.mean(dim=0)

            # L2 drift: ||delta_i - mean_delta||
            diffs = D - mean_delta.unsqueeze(0)
            l2 = torch.norm(diffs, p=2, dim=1)  # [K]

            # cosine similarity: cos(delta_i, mean_delta)
            md_norm = torch.norm(mean_delta, p=2) + 1e-12
            d_norm = torch.norm(D, p=2, dim=1) + 1e-12
            cos = (D @ mean_delta) / (d_norm * md_norm)  # [K]

            drift_mean = float(l2.mean().item())
            drift_std = float(l2.std(unbiased=False).item())
            cos_mean = float(cos.mean().item())
            cos_std = float(cos.std(unbiased=False).item())
        else:
            drift_mean, drift_std, cos_mean, cos_std = 0.0, 0.0, 0.0, 0.0

        # ---- log ----
        history["round"].append(float(r))
        history["test_acc"].append(float(acc))
        history["comm_round_mb"].append(comm_round_mb)
        history["comm_cum_mb"].append(float(comm_cum_mb))
        history["drift_mean_l2"].append(drift_mean)
        history["drift_std_l2"].append(drift_std)
        history["cosine_mean"].append(cos_mean)
        history["cosine_std"].append(cos_std)

        print(
            f"Round {r:02d}/{rounds} | "
            f"test_acc={acc:.4f} | "
            f"comm_cum_mb={comm_cum_mb:.2f} | "
            f"drift_l2={drift_mean:.4f}±{drift_std:.4f} | "
            f"cos={cos_mean:.4f}±{cos_std:.4f}"
        )

    # Save metrics
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)