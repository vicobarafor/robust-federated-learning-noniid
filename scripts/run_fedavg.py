from __future__ import annotations

import os
import json
import random
from typing import Dict, Any, List

import torch
import yaml

from src.data.mnist import get_mnist_dataloaders
from src.models.mnist_cnn import MNISTCNN
from src.training.centralized import evaluate
from src.fl.partition import make_client_loaders
from src.fl.fedavg import local_train, average_weights

# If you have plotting.py and want plots, keep this.
# If you don't want plots, you can delete these 2 lines and the call below.
from src.utils.plotting import save_training_plots


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- dataset config ----
    dcfg = cfg["dataset"]
    batch_size = int(dcfg.get("batch_size", 64))
    eval_batch_size = int(dcfg.get("eval_batch_size", 256))

    # ---- federated config ----
    fcfg = cfg.get("federated", cfg.get("federated", {}))  # keeps you safe if naming changes
    # (your configs use "federated")
    fcfg = cfg["federated"]
    num_clients = int(fcfg["num_clients"])
    clients_per_round = int(fcfg["clients_per_round"])
    rounds = int(fcfg["rounds"])
    local_epochs = int(fcfg["local_epochs"])
    lr = float(fcfg["lr"])
    mu = float(fcfg.get("mu", 0.0))  # mu=0 => FedAvg, mu>0 => FedProx (if supported)

    # ---- logging ----
    out_dir = cfg.get("logging", {}).get("out_dir", "results/run_fedavg")
    os.makedirs(out_dir, exist_ok=True)

    # Save config snapshot
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # ---- load dataset ----
    train_loader, test_loader = get_mnist_dataloaders(batch_size, eval_batch_size)
    full_train_dataset = train_loader.dataset

    # ---- partition config ----
    pcfg = cfg.get("partition", {"type": "iid"})

    # Create client loaders (iid / noniid shards / dirichlet)
    client_loaders = make_client_loaders(
        dataset=full_train_dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        partition_cfg=pcfg,
        seed=seed,
    )

    # ---- global model ----
    global_model = MNISTCNN().to(device)

    history: Dict[str, List[float]] = {"round": [], "test_acc": []}

    for r in range(1, rounds + 1):
        selected_clients = random.sample(range(num_clients), clients_per_round)

        local_states = []

        for cid in selected_clients:
            local_model = MNISTCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())

            # If your local_train supports FedProx, we pass mu + global_model.
            # If your local_train ignores mu, it will still work as FedAvg.
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

        # aggregate
        new_global_state = average_weights(local_states)
        global_model.load_state_dict(new_global_state)

        # evaluate
        test_acc = evaluate(global_model, test_loader, device)
        history["round"].append(float(r))
        history["test_acc"].append(float(test_acc))

        print(f"Round {r:02d}/{rounds} | test_acc={test_acc:.4f}")

    # save metrics
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    # plots (loss/acc curves if your plotting expects these keys; if not, remove this line)
    save_training_plots(history=history, out_dir=out_dir)

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    main(args.config)