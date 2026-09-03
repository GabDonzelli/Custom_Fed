"""Tests for the pretrained Stack Exchange task's pure logic.

Deliberately avoids the network, the 116 MB scan cache and the pretrained
weights, so the suite stays fast and runs on a machine with no internet access
(see the vault note on running this on the HPC cluster).
"""

import torch
from torch import nn

from pytorchexample.tasks.lora import inject_lora, lora_state_dict
from pytorchexample.tasks.stackexchange_pretrained import (
    BLOCK_SIZE,
    IGNORE_INDEX,
    MIN_BLOCK_TOKENS,
    StackExchangePretrainedTask,
    cap_blocks,
    chunk_into_blocks,
)

PAD_ID = 50256


def test_an_exact_multiple_needs_no_padding():
    token_ids = list(range(1, 2 * BLOCK_SIZE + 1))

    blocks = chunk_into_blocks(token_ids, pad_id=PAD_ID)

    assert len(blocks) == 2
    for input_ids, labels in blocks:
        assert len(input_ids) == BLOCK_SIZE
        assert input_ids == labels
        assert IGNORE_INDEX not in labels


def test_chunking_keeps_tokens_that_truncation_would_discard():
    """The from-scratch task truncates and loses 71% of the corpus; this does not."""
    token_ids = list(range(1, 3 * BLOCK_SIZE + 1))

    blocks = chunk_into_blocks(token_ids, pad_id=PAD_ID)

    recovered = [token for input_ids, _ in blocks for token in input_ids]
    assert recovered == token_ids


def test_a_short_tail_is_padded_and_masked_out_of_the_labels():
    tail_length = MIN_BLOCK_TOKENS + 1
    token_ids = list(range(1, BLOCK_SIZE + tail_length + 1))

    blocks = chunk_into_blocks(token_ids, pad_id=PAD_ID)

    assert len(blocks) == 2
    input_ids, labels = blocks[1]
    assert len(input_ids) == BLOCK_SIZE
    assert len(labels) == BLOCK_SIZE
    # Real tokens survive in both; padding is PAD_ID in the input and ignored
    # in the labels, so it never contributes to the loss.
    assert input_ids[:tail_length] == labels[:tail_length]
    assert set(input_ids[tail_length:]) == {PAD_ID}
    assert set(labels[tail_length:]) == {IGNORE_INDEX}


def test_a_tail_below_the_minimum_is_dropped():
    token_ids = list(range(1, BLOCK_SIZE + MIN_BLOCK_TOKENS))

    blocks = chunk_into_blocks(token_ids, pad_id=PAD_ID)

    assert len(blocks) == 1


def test_an_answer_shorter_than_the_minimum_yields_nothing():
    assert chunk_into_blocks(list(range(MIN_BLOCK_TOKENS - 1)), pad_id=PAD_ID) == []


def test_no_tokens_yields_no_blocks():
    assert chunk_into_blocks([], pad_id=PAD_ID) == []


class TinyAdapted(nn.Module):
    """Minimal stand-in for the adapted DistilGPT-2, to test delegation only."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_attn = nn.Linear(8, 24)


def _adapted_model() -> nn.Module:
    model = TinyAdapted()
    inject_lora(model, ("attn.c_attn",), rank=4)
    return model


def test_only_adapters_are_offered_for_communication():
    """The whole point: the frozen base must never reach the wire."""
    task = StackExchangePretrainedTask()
    model = _adapted_model()

    arrays = task.get_federated_arrays(model)

    assert set(arrays) == {"attn.c_attn.lora_a", "attn.c_attn.lora_b"}
    assert all("base" not in name for name in arrays)


def test_communicated_adapters_round_trip_through_the_task():
    task = StackExchangePretrainedTask()

    torch.manual_seed(0)
    source = _adapted_model()
    with torch.no_grad():
        source.attn.c_attn.lora_b.normal_()

    torch.manual_seed(1)
    destination = _adapted_model()
    task.load_federated_arrays(destination, task.get_federated_arrays(source))

    for key, tensor in lora_state_dict(source).items():
        assert torch.equal(lora_state_dict(destination)[key], tensor)


def test_shifted_loss_ignores_padded_positions():
    """A block that is entirely padding after the first token adds no signal."""
    vocab_size = 7
    logits = torch.zeros(1, 4, vocab_size)
    labels = torch.tensor([[3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]])

    loss, predictions, targets = StackExchangePretrainedTask._shifted_loss(
        logits, labels
    )

    # Every shifted target is masked, so cross_entropy has nothing to average.
    assert (targets == IGNORE_INDEX).all()
    assert torch.isnan(loss) or loss.item() == 0.0
    assert predictions.shape == targets.shape


def test_shifted_loss_aligns_prediction_with_the_next_token():
    vocab_size = 5
    labels = torch.tensor([[1, 2, 3, 4]])
    logits = torch.zeros(1, 4, vocab_size)
    # Make position i predict i+2 confidently, which is the label at i+1.
    for position in range(3):
        logits[0, position, position + 2] = 10.0

    _, predictions, targets = StackExchangePretrainedTask._shifted_loss(logits, labels)

    assert targets.tolist() == [[2, 3, 4]]
    assert predictions.tolist() == [[2, 3, 4]]


def _blocks(count: int) -> list[tuple[list[int], list[int]]]:
    return [([index] * BLOCK_SIZE, [index] * BLOCK_SIZE) for index in range(count)]


def test_cap_returns_everything_when_under_the_limit():
    blocks = _blocks(5)
    assert cap_blocks(blocks, limit=10, seed=1) is blocks


def test_cap_returns_everything_when_no_limit_is_set():
    blocks = _blocks(5)
    assert cap_blocks(blocks, limit=None, seed=1) is blocks


def test_cap_subsamples_down_to_the_limit():
    capped = cap_blocks(_blocks(100), limit=7, seed=1)
    assert len(capped) == 7


def test_cap_is_deterministic_for_a_given_seed():
    """A client must see the same local data in every round, not a new sample."""
    first = cap_blocks(_blocks(100), limit=10, seed=42)
    second = cap_blocks(_blocks(100), limit=10, seed=42)
    assert first == second


def test_cap_differs_across_seeds():
    """Different clients subsample differently, so seeds must actually matter."""
    first = cap_blocks(_blocks(100), limit=10, seed=1)
    second = cap_blocks(_blocks(100), limit=10, seed=2)
    assert first != second


def test_cap_preserves_block_contents():
    capped = cap_blocks(_blocks(50), limit=5, seed=3)
    for input_ids, labels in capped:
        assert len(input_ids) == BLOCK_SIZE
        assert input_ids == labels
