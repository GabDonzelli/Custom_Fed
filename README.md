# Grouped Federated Averaging with Flower and PyTorch

This project starts from Flower's PyTorch quickstart and implements two-level federated averaging:

1. Client models are aggregated inside four configured partition groups.
2. The four temporary group models are aggregated proportionally to each group's total number of training examples.

With proportional weights at both levels, the final model is mathematically equivalent to standard example-weighted FedAvg.

The explicit group structure makes it possible to inspect each group and later test alternative group weighting policies.

## Project Structure

```text
pytorchexample/
├── client_app.py
├── server_app.py
├── tasks/
│   ├── base.py
│   ├── cifar10.py
│   └── registry.py
└── strategy/
    ├── aggregation.py
    ├── grouped_fedavg.py
    └── grouping.py
```

- `client_app.py` contains generic Flower client handlers.
- `server_app.py` configures and starts the experiment.
- `tasks/` isolates dataset, model, training, and evaluation details.
- `strategy/` contains grouping and two-level aggregation logic.

## Current Task

CIFAR-10 is the first implemented task.

The task interface is intentionally independent of images, so Shakespeare or a Stack Exchange task can be added by implementing the same methods and registering the new task in:

```text
pytorchexample/tasks/registry.py
```

All models participating in one run must have the same architecture.

The task can change between runs, but models from different tasks cannot be averaged together.

## Configure Partition Groups

The default configuration uses ten partitions and four groups:

```toml
num-partitions = 10
num-groups = 4
partition-groups = "0,1,2|3,4,5|6,7|8,9"
```

The values are zero-based partition IDs, not Flower node IDs.

Every expected partition must appear exactly once.

## Run the Application

Install the project and its dependencies:

```bash
pip install -e .
```

Run the application with a federation containing ten SuperNodes:

```bash
flwr run . --stream
```

The number of SuperNodes must match `num-partitions`.

---

## Technologies

- Python
- PyTorch
- Flower
- Flower Datasets
- CIFAR-10

---

## Current Status

- [x] CIFAR-10 task
- [x] Dataset partition grouping
- [x] Intra-group weighted aggregation
- [x] Inter-group weighted aggregation
- [x] Flower integration
- [x] Standard FedAvg equivalence
- [ ] Alternative group weighting policies
- [ ] Additional datasets
- [ ] Energy-aware aggregation strategies