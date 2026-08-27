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
│   ├── stackexchange.py
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

## Current Tasks

Select the task with `task-name` in `pyproject.toml`:

```toml
task-name = "cifar10"   # or "stackexchange"
```

All models participating in one run must have the same architecture. The
task can change between runs, but models from different tasks cannot be
averaged together.

The task interface (`pytorchexample/tasks/base.py`) is intentionally
independent of any one dataset, so a new task can be added by implementing
the same methods and registering it in `pytorchexample/tasks/registry.py`.

### `cifar10`

IID image classification (CIFAR-10, IID partitioning across clients). The
default task.

### `stackexchange`

Non-IID next-word prediction over `HuggingFaceH4/stack-exchange-preferences`:
each simulated client is one real Stack Exchange author (different people
naturally write about different topics/sites, so this is genuinely non-IID —
unlike CIFAR-10's IID partitioner). A small word-level LSTM predicts the next
token given previous tokens in an answer.

The Hub copy of this dataset does not support cheap row-slicing — even a
`datasets.load_dataset(..., split="train[:2000]")` call downloads full,
multi-gigabyte parquet shards. `pytorchexample/tasks/stackexchange.py` reads
it through the Hugging Face **streaming** API instead and stops after
scanning `MAX_QUESTIONS_SCANNED` questions (default `20_000`), which keeps
the download small and bounded regardless of `num-partitions`.

The `num_partitions` most active qualifying authors (>= 2 answers each,
`MIN_ANSWERS_PER_AUTHOR`) found in that scan become the simulated clients;
`HELD_OUT_AUTHORS` further authors (default `20`) are reserved for
centralized (ServerApp-side) evaluation only, so that measures
generalization to unseen authors. If the scan doesn't find enough qualifying
authors for the configured `num-partitions`, `load_partition_data` /
`load_centralized_data` raise a clear error telling you to raise
`MAX_QUESTIONS_SCANNED` in the source file — these constants (plus `SEQ_LEN`,
`VOCAB_SIZE`) are not exposed as `pyproject.toml` config keys, so adjust them
by editing `pytorchexample/tasks/stackexchange.py` directly.

## Configure Partition Groups

`num-partitions` can be set to any number of simulated clients. Groups are built
automatically according to `group-mode`:

```toml
num-partitions = 100
num-groups = 4
group-mode = "sequential"   # "sequential" | "random" | "manual"
```

- `"sequential"` (default): contiguous ranges, e.g. 100 partitions / 4 groups
  produces `0-24, 25-49, 50-74, 75-99`.
- `"random"`: same group sizes as `"sequential"`, but partitions are shuffled
  into them. Set `group-seed` (an integer) for a reproducible shuffle;
  omit it for a different random split on every run.
- `"manual"`: uses the exact `partition-groups` string, e.g.
  `"0,1,2|3,4,5|6,7|8,9"` (zero-based partition IDs, comma-separated within
  a group, groups separated by `|`). Every expected partition must appear
  exactly once.

Group sizes differ by at most one partition when `num_partitions` doesn't
divide evenly by `num_groups`.

## Partial Client Participation

By default every connected client trains every round (`fraction-train = 1.0`).
To sample only a fraction of clients per round (useful with a large
`num-partitions`):

```toml
fraction-train = 0.1     # ~10% of clients train each round
min-train-nodes = 2      # floor below which fraction-train is ignored
```

At the end of training, the server logs how many rounds each partition
actually participated in:

```
INFO: partition 0: participated in 7/30 training rounds
INFO: partition 1: participated in 8/30 training rounds
...
```

Note: every group must receive at least one reply each round, or that round
raises an error. With a very low `fraction-train` and many groups, it's
possible for a round to sample no client from some group — pick
`fraction-train` / `min-train-nodes` with the number of groups in mind.

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
- [x] Additional datasets (Stack Exchange, non-IID by author)
- [ ] Alternative group weighting policies
- [ ] Energy-aware aggregation strategies