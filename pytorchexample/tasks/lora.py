"""Low-rank adaptation (LoRA) of a frozen pretrained model.

Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022).

Federated fine-tuning of a pretrained model is bounded by communication, not by
compute: ``client_app.py`` ships a whole ``state_dict`` every round, and
DistilGPT-2 is 82M parameters -- 328 MB per client per round. LoRA freezes the
pretrained weights and trains only a low-rank update ``B @ A`` beside each
targeted projection, so only those few adapter tensors have to travel.

``B`` is initialized to zeros, which makes the adapted model *exactly* equal to
the pretrained model before the first optimizer step. That property is what lets
round 0 of a federated run be read as the pretrained model's zero-shot score.

Implemented directly rather than via ``peft`` to keep the dependency footprint
small, matching how ``tf_example.py`` avoids a TensorFlow dependency.
"""

import torch
from torch import nn

# Marks the adapter tensors inside a state_dict. Every LoRA parameter name
# contains this, and nothing else in a pretrained model does, so it is also the
# filter used to decide what gets communicated.
LORA_KEY_MARKER = "lora_"


class LoRALinear(nn.Module):
    """Wrap a frozen projection with a trainable low-rank update.

    Works for both ``nn.Linear`` (weight laid out as ``(out, in)``) and
    HuggingFace's ``Conv1D`` (laid out as ``(in, out)``), because the update is
    applied through explicit matrix multiplications rather than by touching the
    base layer's weight.
    """

    def __init__(
        self,
        base: nn.Module,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank
        # A is randomly initialized and B is zero, so B @ A starts at exactly
        # zero: the wrapped module is initially indistinguishable from `base`.
        self.lora_a = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add the low-rank update to the frozen base layer's output."""
        update = (inputs @ self.lora_a.T) @ self.lora_b.T
        return self.base(inputs) + update * self.scaling


def _projection_shape(module: nn.Module) -> tuple[int, int] | None:
    """Return (in_features, out_features) for a 2-D projection, else None."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    # HuggingFace Conv1D: weight is (in_features, out_features) and `nf` is the
    # output width. Detected structurally so transformers need not be imported.
    weight = getattr(module, "weight", None)
    if weight is not None and weight.dim() == 2 and hasattr(module, "nf"):
        return weight.shape[0], weight.shape[1]
    return None


def inject_lora(
    model: nn.Module,
    target_suffixes: tuple[str, ...],
    rank: int = 8,
    alpha: float = 16.0,
) -> int:
    """Freeze `model` and wrap every module whose name ends in a target suffix.

    Returns the number of modules wrapped. Raises if none matched, since that
    silently produces a model with nothing to train.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(target_suffixes) and _projection_shape(module) is not None
    ]
    if not targets:
        raise ValueError(
            f"No modules matched target suffixes {target_suffixes}. "
            "Check the suffixes against the model's named_modules()."
        )

    for name, module in targets:
        in_features, out_features = _projection_shape(module)
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attribute,
            LoRALinear(module, in_features, out_features, rank=rank, alpha=alpha),
        )

    return len(targets)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only the adapter tensors -- what a client sends to the server."""
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if LORA_KEY_MARKER in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load adapter tensors back into a model, leaving frozen weights alone.

    Raises if `state` carries anything that is not an adapter tensor, or if the
    adapter tensors it carries do not exactly cover the model's own -- either
    would mean the server and the clients disagree about the model.
    """
    unexpected = [name for name in state if LORA_KEY_MARKER not in name]
    if unexpected:
        raise ValueError(
            f"Refusing to load non-adapter tensors into a LoRA model: {unexpected[:5]}"
        )

    expected = set(lora_state_dict(model))
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(
            "LoRA state does not match the model's adapters "
            f"(missing {missing[:5]}, unexpected {extra[:5]})."
        )

    model.load_state_dict(state, strict=False)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the parameters an optimizer should actually update."""
    return [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
