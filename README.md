# Drift-Aware Adaptive Aggregation (DAA)
### Federated Learning Under Data Heterogeneity

Federated learning systems frequently operate in environments where client datasets are **strongly non-IID**. Under these conditions, local model updates can diverge substantially, and uniform aggregation methods such as FedAvg may amplify instability rather than correct it.

This repository investigates that phenomenon on **CIFAR-10** and introduces a simple aggregation mechanism called **Drift-Aware Adaptive Aggregation (DAA)**.

Instead of treating all client updates equally, DAA adjusts aggregation weights according to **how far each client update deviates from the mean update direction** during each communication round.

The result is an aggregation rule that remains lightweight while exposing useful diagnostics about update drift and heterogeneity.

---

# Main Experimental Results

![Main Results](plots/main_results_figure.png)

The figure summarizes the core observations from our experiments:

• Model accuracy as client data becomes increasingly heterogeneous  
• Communication cost relative to final model performance  
• Measured drift of client updates under different partition regimes

---

# Motivation

Federated learning algorithms are typically evaluated under either IID partitions or mild heterogeneity. However, realistic deployments often exhibit **extreme client distribution skew**, particularly when datasets are partitioned by user behavior or device context.

Under these conditions:

- client gradients can point in conflicting directions  
- uniform averaging may overweight noisy or divergent updates  
- model convergence may slow or degrade

This project explores whether **update-distance awareness at the server aggregation stage** can partially mitigate this behavior.

---

# Drift-Aware Adaptive Aggregation

Let the update from client *i* be

Δᵢ = wᵢ − w_global

where *w_global* is the current server model.

Compute the mean update

Δ̄ = (1/K) Σ Δᵢ

DAA assigns a weight to each client update based on its deviation from the mean:

αᵢ ∝ exp(−β ||Δᵢ − Δ̄||)

Clients whose updates drift significantly from the consensus direction receive lower aggregation weight.

The aggregated update becomes

w_next = w_global + Σ αᵢ Δᵢ

This introduces **drift-sensitivity** without modifying client optimization or communication patterns.

---

# Experimental Setup

Dataset  
CIFAR-10

Model  
ResNet architecture for CIFAR classification

Federated configuration

• total clients: 20  
• clients sampled per round: 10  
• communication rounds: 20  
• local training: SGD  

Random seeds

• 1  
• 7  
• 42  

Client partition regimes

| Partition Type | Description |
|---|---|
| IID | uniform distribution across clients |
| Dirichlet α = 0.5 | moderate heterogeneity |
| Dirichlet α = 0.3 | strong heterogeneity |
| Dirichlet α = 0.1 | extreme heterogeneity |

Methods compared

• FedAvg  
• FedProx  
• Drift-Aware Adaptive Aggregation (DAA)

---

# Accuracy Under Increasing Heterogeneity

![Accuracy](plots/accuracy_comparison.png)

The accuracy curve illustrates how performance degrades as the Dirichlet concentration parameter decreases.

Key observations:

• IID partitions achieve the highest accuracy across all methods  
• moderate heterogeneity produces only mild degradation  
• extreme heterogeneity (α = 0.1) significantly reduces performance  
• DAA provides additional visibility into drift statistics during training

---

# Communication Efficiency

![Communication](plots/communication_efficiency.png)

All methods operate under identical communication budgets.

Observations:

• FedAvg and FedProx achieve slightly higher peak accuracy  
• DAA achieves competitive performance with **significantly lower communication cost** due to fewer rounds in this experiment configuration

---

# Measured Update Drift

![Drift](plots/daa_drift_vs_heterogeneity.png)

The drift metric measures the L2 distance between individual client updates and the mean update vector.

Observed pattern:

• drift increases sharply as heterogeneity increases  
• IID partitions produce minimal update divergence  
• extreme Dirichlet partitions produce the highest update variance

This provides empirical evidence that **update drift correlates strongly with dataset heterogeneity**.

---

# Aggregation Diagnostics

Unlike traditional federated algorithms, DAA exposes several aggregation statistics:

• aggregation weight variance  
• update distance statistics  
• drift magnitude across clients  

These diagnostics allow inspection of **how the server interprets client updates during training**, not just the final accuracy.

---

# Repository Structure

src/
  fl/
    daa.py
    fedavg.py
    partition.py

scripts/
  run_daa_cifar10.py

configs/
  cifar10/
    daa_cifar10_*.yaml

plots/
  main_results_figure.png
  accuracy_comparison.png
  communication_efficiency.png
  daa_drift_vs_heterogeneity.png

results/
  summary_results.csv
  final_comparison_table.csv

---

# Running Experiments

Example run

python -m scripts.run_daa_cifar10 --config configs/cifar10/daa_cifar10_a03_s1.yaml

Each configuration writes results to a separate directory to avoid collisions across seeds.

---

# Reproducibility

All experiments were executed using:

• three independent random seeds  
• identical model architecture  
• identical communication budgets

Results are stored as `metrics.json` files and aggregated during analysis to produce the final comparison tables and plots.

---

# Discussion

These experiments suggest that **client update drift is a measurable and informative signal in heterogeneous federated learning**.

While DAA does not always outperform strong baselines in raw accuracy, it provides a simple mechanism for **detecting and attenuating extreme client updates**, which may be valuable in highly skewed or unreliable environments.

Future work could explore:

• theoretical analysis of drift-weighted aggregation  
• robustness under adversarial clients  
• scaling to larger client populations  
• integration with secure aggregation protocols

---

# Author Note

This repository was developed as an experimental investigation into **aggregation behavior under heterogeneous federated training**. The project emphasizes reproducibility, diagnostic metrics, and transparent experiment pipelines rather than single-metric benchmark optimization.
