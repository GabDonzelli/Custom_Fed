"""End-to-end smoke test of the pretrained task, including the Flower wire path."""

import torch
from flwr.app import ArrayRecord

from pytorchexample.tasks.registry import get_task

NUM_PARTITIONS = 3
DEVICE = torch.device("cpu")

task = get_task("stackexchange-pretrained")

print("create_model...", flush=True)
model = task.create_model()
total = sum(p.numel() for p in model.parameters())
arrays = task.get_federated_arrays(model)
sent = sum(t.numel() for t in arrays.values())
print(f"  total {total:,} params; communicating {sent:,} ({100*sent/total:.3f}%)", flush=True)

# The actual wire path: dict -> ArrayRecord -> back to a fresh model.
print("ArrayRecord round-trip...", flush=True)
record = ArrayRecord(arrays)
restored = record.to_torch_state_dict()
fresh = task.create_model()
task.load_federated_arrays(fresh, restored)
for key, tensor in arrays.items():
    assert torch.equal(task.get_federated_arrays(fresh)[key], tensor), key
print(f"  OK, {len(restored)} tensors survived serialization", flush=True)

print("load_partition_data (loads the 116MB scan cache)...", flush=True)
trainloader, validationloader = task.load_partition_data(
    partition_id=0, num_partitions=NUM_PARTITIONS, batch_size=4
)
print(f"  train blocks {len(trainloader.dataset):,}, val blocks {len(validationloader.dataset):,}", flush=True)

print("evaluate BEFORE training (this is the zero-shot pretrained model)...", flush=True)
before = task.evaluate(model, validationloader, DEVICE)
print(f"  {before}", flush=True)

print("train 1 epoch on a few batches...", flush=True)
subset = torch.utils.data.DataLoader(
    torch.utils.data.Subset(trainloader.dataset, range(min(24, len(trainloader.dataset)))),
    batch_size=4,
    shuffle=True,
)
metrics = task.train(model, subset, epochs=1, learning_rate=5e-4, device=DEVICE)
print(f"  {metrics}", flush=True)

print("evaluate AFTER training...", flush=True)
after = task.evaluate(model, validationloader, DEVICE)
print(f"  {after}", flush=True)

print(f"\naccuracy {before['accuracy']:.4f} -> {after['accuracy']:.4f}", flush=True)
print(f"loss     {before['loss']:.4f} -> {after['loss']:.4f}", flush=True)

# After a step, the adapters must have moved off zero.
b_norms = [
    t.norm().item() for name, t in task.get_federated_arrays(model).items() if "lora_b" in name
]
print(f"lora_b norms after training (must be > 0): {max(b_norms):.4e}", flush=True)
