from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt


def save_training_plots(history: Dict[str, List[float]], out_dir: str) -> None:
    """
    Robust plotting:
    - Always plot test_acc if present
    - Plot train_loss/test_loss only if those keys exist
    - Plot train_acc/test_acc together if both exist
    """
    os.makedirs(out_dir, exist_ok=True)

    # X-axis: prefer round, fallback to epoch, fallback to index
    if "round" in history:
        x = history["round"]
        x_label = "Round"
    elif "epoch" in history:
        x = history["epoch"]
        x_label = "Epoch"
    else:
        # fallback
        any_key = next(iter(history.keys()))
        x = list(range(1, len(history[any_key]) + 1))
        x_label = "Step"

    # ---- Accuracy plot (most important for FL runs) ----
    if "test_acc" in history or "train_acc" in history:
        plt.figure()
        if "train_acc" in history:
            plt.plot(x, history["train_acc"], label="Train acc")
        if "test_acc" in history:
            plt.plot(x, history["test_acc"], label="Test acc")
        plt.xlabel(x_label)
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
        plt.close()

    # ---- Loss plot (only if available) ----
    if "train_loss" in history or "test_loss" in history:
        plt.figure()
        if "train_loss" in history:
            plt.plot(x, history["train_loss"], label="Train loss")
        if "test_loss" in history:
            plt.plot(x, history["test_loss"], label="Test loss")
        plt.xlabel(x_label)
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"))
        plt.close()