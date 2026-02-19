import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import json
import os

import torch
import yaml

from src.data.mnist import get_mnist_dataloaders
from src.models.mnist_cnn import MNISTCNN
from src.training.centralized import train_centralized
from src.utils.plotting import save_training_plots

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["lr"])
    eval_batch_size = int(cfg["eval"]["batch_size"])

    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train_loader, test_loader = get_mnist_dataloaders(batch_size, eval_batch_size)

    model = MNISTCNN()
    history = train_centralized(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
    )

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Saved results to: {out_dir}")
    save_training_plots(history, out_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    main(args.config)