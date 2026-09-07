"""Tests for the frozen-backbone CIFAR-10 task.

Nothing here downloads CIFAR-10 or the ImageNet weights: the parts worth
testing are the ones that would silently corrupt a run -- what gets federated,
and whether the feature normalization is client-independent.
"""

import torch

from pytorchexample.tasks.cifar10_pretrained import (
    FEATURE_DIM,
    NUM_CLASSES,
    Cifar10PretrainedTask,
    l2_normalize,
)


def test_only_the_linear_head_is_federated():
    """5,130 floats, not the 11.2M of a ResNet-18."""
    task = Cifar10PretrainedTask()
    model = task.create_model()
    arrays = task.get_federated_arrays(model)

    assert set(arrays) == {"classifier.weight", "classifier.bias"}
    assert sum(tensor.numel() for tensor in arrays.values()) == (
        FEATURE_DIM * NUM_CLASSES + NUM_CLASSES
    )


def test_federated_arrays_round_trip():
    """A head loaded back into a fresh model must be the same head."""
    task = Cifar10PretrainedTask()
    source = task.create_model()
    target = task.create_model()
    task.load_federated_arrays(target, task.get_federated_arrays(source))

    for source_tensor, target_tensor in zip(
        source.state_dict().values(), target.state_dict().values()
    ):
        assert torch.equal(source_tensor, target_tensor)


def test_normalization_is_per_sample():
    """Each row is scaled by its own norm, so no client-wide statistic leaks in.

    This is the property that keeps FedAvg meaningful: were the features
    standardized with a client's own mean and variance, two clients' heads
    would describe different input spaces and averaging them would be
    arithmetic without meaning.
    """
    features = torch.randn(8, FEATURE_DIM)
    normalized = l2_normalize(features)

    assert torch.allclose(normalized.norm(dim=1), torch.ones(8), atol=1e-5)

    # Normalizing a row alone gives the same answer as normalizing it in a
    # batch: the transform sees no other sample.
    alone = l2_normalize(features[3:4])
    assert torch.allclose(alone[0], normalized[3], atol=1e-6)


def test_normalization_survives_a_zero_vector():
    """A dead feature vector must not become NaN and poison the average."""
    normalized = l2_normalize(torch.zeros(2, FEATURE_DIM))
    assert torch.isfinite(normalized).all()


def test_train_and_evaluate_on_synthetic_features():
    """The head must actually learn from features, and report real metrics."""
    torch.manual_seed(0)
    task = Cifar10PretrainedTask()
    model = task.create_model()

    # Two linearly separable clusters labelled 0 and 1.
    centers = torch.zeros(2, FEATURE_DIM)
    centers[0, 0] = 1.0
    centers[1, 1] = 1.0
    features = l2_normalize(
        centers.repeat_interleave(32, dim=0) + torch.randn(64, FEATURE_DIM) * 0.01
    )
    labels = torch.tensor([0] * 32 + [1] * 32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size=16, shuffle=True
    )

    device = torch.device("cpu")
    before = task.evaluate(model, loader, device)["accuracy"]
    task.train(model, loader, epochs=30, learning_rate=1.0, device=device)
    after = task.evaluate(model, loader, device)

    assert after["accuracy"] > before
    assert after["accuracy"] > 0.9
    assert after["loss"] > 0.0
