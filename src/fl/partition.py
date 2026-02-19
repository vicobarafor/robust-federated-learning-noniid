from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
from torch.utils.data import DataLoader, Subset


def _get_targets(dataset) -> np.ndarray:
    """
    Robustly extract labels from common torchvision datasets (MNIST/CIFAR/etc.)
    """
    if hasattr(dataset, "targets"):
        t = dataset.targets
        # torch.Tensor or list -> numpy
        try:
            return t.detach().cpu().numpy()
        except Exception:
            return np.array(t)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    if hasattr(dataset, "train_labels"):
        t = dataset.train_labels
        try:
            return t.detach().cpu().numpy()
        except Exception:
            return np.array(t)
    raise ValueError("Could not find labels/targets on dataset. Add a custom extractor for your dataset.")


def iid_partition(num_samples: int, num_clients: int, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    all_idx = np.arange(num_samples)
    rng.shuffle(all_idx)
    parts = np.array_split(all_idx, num_clients)
    return [p.astype(int) for p in parts]


def shard_partition(dataset, num_clients: int, shards_per_client: int, seed: int = 42) -> List[np.ndarray]:
    """
    Classic label-shard non-IID:
    - sort by label
    - cut into shards
    - give each client `shards_per_client` shards
    """
    rng = np.random.default_rng(seed)
    targets = _get_targets(dataset)
    n = len(targets)

    # Sort indices by label
    idx = np.arange(n)
    idx_sorted = idx[np.argsort(targets)]

    num_shards = num_clients * shards_per_client
    if num_shards > n:
        raise ValueError(f"Too many shards ({num_shards}) for dataset size {n}. Reduce shards_per_client or num_clients.")

    shards = np.array_split(idx_sorted, num_shards)
    shard_ids = np.arange(num_shards)
    rng.shuffle(shard_ids)

    parts: List[np.ndarray] = []
    for c in range(num_clients):
        chosen = shard_ids[c * shards_per_client : (c + 1) * shards_per_client]
        client_idx = np.concatenate([shards[sid] for sid in chosen], axis=0)
        rng.shuffle(client_idx)
        parts.append(client_idx.astype(int))
    return parts


def dirichlet_partition(dataset, num_clients: int, alpha: float, seed: int = 42) -> List[np.ndarray]:
    """
    Dirichlet label-skew partition:
    For each class k: distribute its indices across clients with Dir(alpha).
    Smaller alpha => more non-IID.
    """
    if alpha <= 0:
        raise ValueError("dirichlet alpha must be > 0")

    rng = np.random.default_rng(seed)
    targets = _get_targets(dataset)
    n = len(targets)
    classes = np.unique(targets)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for k in classes:
        idx_k = np.where(targets == k)[0]
        rng.shuffle(idx_k)

        # proportions over clients
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # turn proportions into counts
        counts = (proportions * len(idx_k)).astype(int)

        # fix rounding so total matches
        diff = len(idx_k) - counts.sum()
        # distribute remainder
        for i in rng.choice(num_clients, size=abs(diff), replace=True):
            counts[i] += 1 if diff > 0 else -1

        # split idx_k by counts
        start = 0
        for c in range(num_clients):
            cnt = int(counts[c])
            if cnt > 0:
                client_indices[c].extend(idx_k[start : start + cnt].tolist())
            start += cnt

    parts = []
    for c in range(num_clients):
        arr = np.array(client_indices[c], dtype=int)
        rng.shuffle(arr)
        parts.append(arr)

    # Safety: ensure we didn't lose indices completely (may happen if alpha extremely tiny)
    if sum(len(p) for p in parts) == 0:
        raise RuntimeError("Dirichlet partition produced empty allocation. Try a larger alpha.")

    return parts


def make_client_loaders(
    dataset,
    num_clients: int,
    batch_size: int,
    partition_cfg: Dict[str, Any],
    seed: int = 42,
):
    """
    partition_cfg examples:
      {"type": "iid"}
      {"type": "noniid", "shards_per_client": 2}
      {"type": "dirichlet", "dirichlet_alpha": 0.1}
    """
    ptype = partition_cfg.get("type", "iid")

    if ptype == "iid":
        parts = iid_partition(len(dataset), num_clients, seed=seed)

    elif ptype == "noniid":
        spc = int(partition_cfg.get("shards_per_client", 2))
        parts = shard_partition(dataset, num_clients=num_clients, shards_per_client=spc, seed=seed)

    elif ptype == "dirichlet":
        alpha = float(partition_cfg.get("dirichlet_alpha", 0.1))
        parts = dirichlet_partition(dataset, num_clients=num_clients, alpha=alpha, seed=seed)

    else:
        raise ValueError(f"Unknown partition.type: {ptype} (expected iid | noniid | dirichlet)")

    loaders = []
    for idx in parts:
        subset = Subset(dataset, idx.tolist())
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)

    return loaders