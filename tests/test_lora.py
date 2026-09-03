"""Tests for the LoRA adapter implementation.

None of these need a pretrained model: LoRA's contract is about shapes, freezing
and which tensors travel, all of which are exercised with tiny local modules.
"""

import pytest
import torch
from torch import nn

from pytorchexample.tasks.lora import (
    LORA_KEY_MARKER,
    LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    trainable_parameters,
)


class Conv1DLike(nn.Module):
    """Stand-in for HuggingFace's Conv1D: weight is (in_features, out_features).

    Written out here so the (in, out) layout -- the opposite of nn.Linear -- is
    covered without importing transformers.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.nf = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight + self.bias


class TinyBlock(nn.Module):
    """Two-projection model whose submodule names mimic a transformer block."""

    def __init__(self, conv1d_style: bool = False) -> None:
        super().__init__()
        projection = Conv1DLike if conv1d_style else nn.Linear
        self.attn = nn.Module()
        self.attn.c_attn = projection(8, 24)
        self.attn.c_proj = projection(8, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.attn.c_attn(inputs)


@pytest.mark.parametrize("conv1d_style", [False, True])
def test_injection_is_numerically_a_no_op(conv1d_style):
    """B starts at zero, so the wrapped model must match the original exactly."""
    torch.manual_seed(0)
    model = TinyBlock(conv1d_style=conv1d_style)
    inputs = torch.randn(4, 8)

    with torch.no_grad():
        before = model(inputs)
    inject_lora(model, ("attn.c_attn",), rank=4)
    with torch.no_grad():
        after = model(inputs)

    assert torch.equal(before, after)


@pytest.mark.parametrize("conv1d_style", [False, True])
def test_adapter_changes_output_once_b_is_nonzero(conv1d_style):
    """Sanity check that the update is actually wired into the forward pass."""
    torch.manual_seed(0)
    model = TinyBlock(conv1d_style=conv1d_style)
    inputs = torch.randn(4, 8)
    inject_lora(model, ("attn.c_attn",), rank=4)

    with torch.no_grad():
        before = model(inputs)
        model.attn.c_attn.lora_b.fill_(0.5)
        after = model(inputs)

    assert not torch.allclose(before, after)


def test_injection_wraps_only_the_targeted_suffix():
    model = TinyBlock()
    wrapped = inject_lora(model, ("attn.c_attn",), rank=4)

    assert wrapped == 1
    assert isinstance(model.attn.c_attn, LoRALinear)
    assert not isinstance(model.attn.c_proj, LoRALinear)


def test_injection_can_target_several_suffixes():
    model = TinyBlock()
    assert inject_lora(model, ("attn.c_attn", "attn.c_proj"), rank=4) == 2


def test_injection_raises_when_nothing_matches():
    """A silent no-match would produce a model with nothing trainable."""
    with pytest.raises(ValueError, match="No modules matched"):
        inject_lora(TinyBlock(), ("does.not.exist",), rank=4)


def test_non_positive_rank_is_rejected():
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRALinear(nn.Linear(4, 4), 4, 4, rank=0)


def test_base_weights_are_frozen_and_only_adapters_train():
    model = TinyBlock()
    inject_lora(model, ("attn.c_attn",), rank=4)

    trainable = trainable_parameters(model)
    assert trainable, "LoRA injection must leave something trainable"
    assert all(LORA_KEY_MARKER in name
               for name, parameter in model.named_parameters()
               if parameter.requires_grad)
    # Two adapter tensors (A and B) for the single wrapped projection.
    assert len(trainable) == 2


def test_adapter_shapes_follow_in_and_out_features():
    model = TinyBlock()
    inject_lora(model, ("attn.c_attn",), rank=4)

    # c_attn maps 8 -> 24, so A is (rank, 8) and B is (24, rank).
    assert model.attn.c_attn.lora_a.shape == (4, 8)
    assert model.attn.c_attn.lora_b.shape == (24, 4)


def test_lora_state_dict_carries_only_adapters():
    model = TinyBlock()
    inject_lora(model, ("attn.c_attn",), rank=4)

    state = lora_state_dict(model)
    assert set(state) == {
        "attn.c_attn.lora_a",
        "attn.c_attn.lora_b",
    }
    # The frozen base is far larger than what actually travels.
    assert len(state) < len(model.state_dict())


def test_adapters_round_trip_through_load():
    torch.manual_seed(0)
    source = TinyBlock()
    inject_lora(source, ("attn.c_attn",), rank=4)
    with torch.no_grad():
        source.attn.c_attn.lora_b.normal_()

    torch.manual_seed(1)
    destination = TinyBlock()
    inject_lora(destination, ("attn.c_attn",), rank=4)

    load_lora_state_dict(destination, lora_state_dict(source))

    for key, tensor in lora_state_dict(source).items():
        assert torch.equal(lora_state_dict(destination)[key], tensor)


def test_load_leaves_frozen_base_weights_untouched():
    """Loading adapters must not overwrite the pretrained weights."""
    torch.manual_seed(0)
    model = TinyBlock()
    inject_lora(model, ("attn.c_attn",), rank=4)
    original_base = model.attn.c_attn.base.weight.clone()

    other = TinyBlock()
    inject_lora(other, ("attn.c_attn",), rank=4)
    with torch.no_grad():
        other.attn.c_attn.lora_b.normal_()
    load_lora_state_dict(model, lora_state_dict(other))

    assert torch.equal(model.attn.c_attn.base.weight, original_base)


def test_load_rejects_a_full_state_dict():
    """Guards against a task accidentally sending the whole model."""
    model = TinyBlock()
    inject_lora(model, ("attn.c_attn",), rank=4)

    with pytest.raises(ValueError, match="non-adapter tensors"):
        load_lora_state_dict(model, dict(model.state_dict()))


def test_load_rejects_a_mismatched_adapter_set():
    """Server and clients disagreeing about the model must fail loudly."""
    model = TinyBlock()
    inject_lora(model, ("attn.c_attn",), rank=4)

    state = lora_state_dict(model)
    state.pop("attn.c_attn.lora_b")
    with pytest.raises(ValueError, match="does not match"):
        load_lora_state_dict(model, state)


def test_gradients_reach_adapters_but_not_the_base():
    model = TinyBlock()
    inject_lora(model, ("attn.c_attn",), rank=4)

    model(torch.randn(4, 8)).sum().backward()

    assert model.attn.c_attn.lora_a.grad is not None
    assert model.attn.c_attn.lora_b.grad is not None
    assert model.attn.c_attn.base.weight.grad is None
