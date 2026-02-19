# Robust Federated Learning Under Non-IID Label Skew

This repository contains a controlled empirical study comparing FedAvg and FedProx under varying degrees of statistical heterogeneity induced by label-skewed data partitions.

The goal is to evaluate algorithm robustness as client data distributions deviate from IID assumptions.

---

## 1. Problem Setting

Federated learning assumes decentralized clients with local data. In practice, client datasets are rarely IID. This project investigates:

- How performance degrades under increasing non-IID label skew.
- Whether FedProx improves stability relative to FedAvg.

---

## 2. Experimental Setup

Dataset:
- MNIST

Model:
- Convolutional neural network (2 convolution layers + fully connected head)

Federated configuration:
- 20 total clients
- 10 clients sampled per round
- 20 communication rounds
- Local SGD optimizer
- μ = 0.1 for FedProx experiments

Partition strategies:
- IID
- Non-IID (1 shard per client)
- Non-IID (2 shards per client)
- Non-IID (3 shards per client)

Non-IID partitions are constructed via label-sorted shard allocation.

---

## 3. Results (Final Test Accuracy)

| Partition | FedAvg | FedProx (μ=0.1) |
|------------|---------|----------------|
| 1 shard    | 0.5614  | 0.6528 (mean over seeds) |
| 2 shards   | 0.8583  | 0.9031 |
| 3 shards   | 0.8889  | 0.9364 |

Under extreme heterogeneity (1 shard), FedAvg suffers substantial degradation.  
FedProx improves stability and average performance.  
As shard count increases, statistical heterogeneity decreases and both methods improve.

---

## 4. Reproducibility

All experiments are reproducible via configuration files in `configs/sweeps/`.

Example:
python -m scripts.run_fedavg --config configs/sweeps/shard1_fedprox_mu01.yaml

## 5. Repository Structure
configs/ Experiment configurations
scripts/ Training runners
src/ Models, partitioning, FL algorithms


---

## 6. Future Extensions

- Extend experiments to CIFAR-10
- Measure gradient divergence between clients
- Analyze communication vs performance trade-offs
- Compare additional robustness-oriented FL algorithms


## Experimental Results

### Final Test Accuracy (20 communication rounds)

| Partition (Label Skew) | Method                     | Final Accuracy |
|------------------------|----------------------------|---------------|
| Shard = 1              | FedAvg                     | 0.5614 |
| Shard = 1              | FedProx (μ = 0.01)         | 0.6528 ± 0.0313 |
| Shard = 2              | FedAvg                     | 0.8583 |
| Shard = 2              | FedProx (μ = 0.01)         | 0.9031 |
| Shard = 3              | FedAvg                     | 0.8889 |
| Shard = 3              | FedProx (μ = 0.01)         | 0.9364 |

---

## Discussion

This study evaluates the behavior of FedAvg and FedProx under controlled label-skew non-IID settings on MNIST.

Under extreme heterogeneity (Shard = 1), FedAvg exhibits instability and degraded performance.  
FedProx improves robustness by regularizing client updates toward the global model.

As heterogeneity decreases (Shard = 2 → 3), the performance gap narrows, but FedProx consistently maintains superior convergence behavior.

These results highlight the importance of proximal regularization in highly non-IID federated environments.

---

## Reproducibility


All experiments are fully reproducible via configuration files.

Example (Shard = 3, FedProx):

```bash
python -m scripts.run_fedavg --config configs/sweeps/shard3_fedprox_mu01.yaml
```

Multi-seed evaluation (Shard = 1):

```bash
python -m scripts.run_fedavg --config configs/sweeps/shard1_fedprox_mu01_seed1.yaml
python -m scripts.run_fedavg --config configs/sweeps/shard1_fedprox_mu01_seed7.yaml
```