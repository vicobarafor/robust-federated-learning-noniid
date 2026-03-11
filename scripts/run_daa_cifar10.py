from __future__ import annotations

import os
import json
import argparse
from copy import deepcopy
from typing import Dict, Any, List

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.resnet_cifar import ResNetCIFAR
from src.fl.partition import make_client_loaders
from src.fl.fedavg import local_train
from src.fl.daa import drift_adaptive_average


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


def _model_num_bytes(model: torch.nn.Module) -> int:
    total = 0
    with torch.no_grad():
        for p in model.parameters():
            total += p.numel() * p.element_size()
        for b in model.buffers():
            total += b.numel() * b.element_size()
    return int(total)


def _cfg_get_partition(cfg: Dict[str, Any]) -> Dict[str, Any]:
    pcfg = cfg.get("partition", {"type": "iid"})
    if isinstance(pcfg, str):
        return {"type": pcfg}
    return pcfg


def _make_run_name(cfg: Dict[str, Any]) -> str:
    for key in ["run_name", "name", "experiment_name"]:
        if key in cfg and cfg[key]:
            return str(cfg[key])
    return "daa_cifar10_run"


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    dcfg = cfg["dataset"]
    batch_size = int(dcfg.get("batch_size", 64))
    eval_batch_size = int(dcfg.get("eval_batch_size", 256))

    train_loader, test_loader = get_cifar10_dataloaders(batch_size, eval_batch_size)
    train_ds = train_loader.dataset

    fcfg = cfg["federated"]
    num_clients = int(fcfg["num_clients"])
    clients_per_round = int(fcfg["clients_per_round"])
    rounds = int(fcfg["rounds"])
    local_epochs = int(fcfg["local_epochs"])
    lr = float(fcfg["lr"])

    method_cfg = cfg.get("method", {})
    beta = float(method_cfg.get("beta", 1.0))

    pcfg = _cfg_get_partition(cfg)
    partition_cfg = {"type": pcfg.get("type", "iid")}
    if partition_cfg["type"] == "dirichlet":
        partition_cfg["dirichlet_alpha"] = pcfg.get("alpha", 0.1)

    client_loaders = make_client_loaders(
        train_ds,
        num_clients=num_clients,
        batch_size=batch_size,
        partition_cfg=partition_cfg,
        seed=seed,
    )

    global_model = ResNetCIFAR(num_classes=10).to(device)
    model_mb = _model_num_bytes(global_model) / (1024.0 * 1024.0)

    run_name = _make_run_name(cfg)
    out_dir = cfg.get("output_dir", f"/content/drive/MyDrive/fl_results/{run_name}")
    os.makedirs(out_dir, exist_ok=True)

    history = {
        "round": [],
        "test_acc": [],
        "comm_cum_mb": [],
        "agg_weight_mean": [],
        "agg_weight_std": [],
        "agg_weight_min": [],
        "agg_weight_max": [],
        "agg_distance_mean": [],
        "agg_distance_std": [],
        "agg_distance_min": [],
        "agg_distance_max": [],
    }

    comm_cum_mb = 0.0

    for rnd in range(1, rounds + 1):
        rng = np.random.default_rng(seed + rnd)
        selected = rng.choice(num_clients, size=clients_per_round, replace=False).tolist()

        global_state = deepcopy(global_model.state_dict())
        local_states: List[Dict[str, torch.Tensor]] = []

        for cid in selected:
            local_model = ResNetCIFAR(num_classes=10).to(device)
            local_model.load_state_dict(global_state)

            sd = local_train(
                model=local_model,
                loader=client_loaders[cid],
                device=device,
                epochs=local_epochs,
                lr=lr,
                mu=0.0,
                global_model=None,
            )
            local_states.append(sd)

        new_state, diag = drift_adaptive_average(
            local_states=local_states,
            global_state=global_state,
            beta=beta,
        )

        global_model.load_state_dict(new_state)
        acc = evaluate(global_model, test_loader, device=device)

        comm_round_mb = float(2.0 * clients_per_round * model_mb)
        comm_cum_mb += comm_round_mb

        history["round"].append(rnd)
        history["test_acc"].append(float(acc))
        history["comm_cum_mb"].append(float(comm_cum_mb))
        history["agg_weight_mean"].append(float(diag.get("weight_mean", float("nan"))))
        history["agg_weight_std"].append(float(diag.get("weight_std", float("nan"))))
        history["agg_weight_min"].append(float(diag.get("weight_min", float("nan"))))
        history["agg_weight_max"].append(float(diag.get("weight_max", float("nan"))))
        history["agg_distance_mean"].append(float(diag.get("distance_mean", float("nan"))))
        history["agg_distance_std"].append(float(diag.get("distance_std", float("nan"))))
        history["agg_distance_min"].append(float(diag.get("distance_min", float("nan"))))
        history["agg_distance_max"].append(float(diag.get("distance_max", float("nan"))))

        print(
            f"Round {rnd:02d}/{rounds:02d} | "
            f"test_acc={acc:.4f} | "
            f"comm_cum_mb={comm_cum_mb:.2f} | "
            f"drift_l2={diag.get('distance_mean', float('nan')):.4f}±{diag.get('distance_std', float('nan')):.4f} | "
            f"wstd={diag.get('weight_std', float('nan')):.4f}",
            flush=True
        )

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Saved results to: {out_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
