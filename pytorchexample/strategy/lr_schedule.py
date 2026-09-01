"""Round-based learning-rate schedules for federated training.

Large-cohort federated training tolerates a large learning rate early on,
but needs a much smaller one late in training for the global model to
converge -- see Charles et al., "On Large-Cohort Training for Federated
Learning" (2021). Each schedule below maps a 1-based server round to the
learning rate broadcast to clients for that round.

The builder returns a pure callable, so a schedule can be tabulated and
unit-tested without running a federation.
"""

import math
from collections.abc import Callable

LRSchedule = Callable[[int], float]

LR_SCHEDULES = ("constant", "cosine", "exponential", "step")


def build_lr_schedule(
    schedule: str,
    initial_lr: float,
    num_rounds: int,
    min_lr: float = 0.0,
    decay_rate: float = 0.99,
    step_size: int = 100,
) -> LRSchedule:
    """Build a round-to-learning-rate callable.

    Every schedule returns ``initial_lr`` for round 1 and never drops below
    ``min_lr``. Rounds below 1 (the initial centralized evaluation uses
    round 0) are treated as round 1.

    Parameters
    ----------
    schedule:
        One of ``LR_SCHEDULES``.
    initial_lr:
        Learning rate used in round 1, before any decay.
    num_rounds:
        Total number of federated rounds. Only ``cosine`` uses this, to
        reach ``min_lr`` exactly at the final round.
    min_lr:
        Floor for the returned learning rate.
    decay_rate:
        Per-round multiplier for ``exponential``, per-step multiplier for
        ``step``. Must be in (0, 1].
    step_size:
        Rounds between drops for ``step``.
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(
            f"Unknown lr-schedule {schedule!r}. Expected one of {LR_SCHEDULES}."
        )
    if initial_lr <= 0.0:
        raise ValueError(f"initial_lr must be positive, but received {initial_lr}.")
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be at least 1, but received {num_rounds}.")
    if min_lr < 0.0:
        raise ValueError(f"min_lr cannot be negative, but received {min_lr}.")
    if min_lr > initial_lr:
        raise ValueError(
            f"min_lr ({min_lr}) cannot exceed initial_lr ({initial_lr})."
        )
    if not 0.0 < decay_rate <= 1.0:
        raise ValueError(
            f"decay_rate must be in (0, 1], but received {decay_rate}."
        )
    if step_size < 1:
        raise ValueError(f"step_size must be at least 1, but received {step_size}.")

    def constant(server_round: int) -> float:
        """Keep the learning rate fixed (the no-decay baseline)."""
        del server_round
        return initial_lr

    def cosine(server_round: int) -> float:
        """Anneal smoothly from initial_lr in round 1 to min_lr in the last."""
        if num_rounds == 1:
            return initial_lr
        progress = _round_index(server_round) / (num_rounds - 1)
        progress = min(progress, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (initial_lr - min_lr) * cosine_factor

    def exponential(server_round: int) -> float:
        """Multiply the learning rate by decay_rate every round."""
        decayed = initial_lr * decay_rate ** _round_index(server_round)
        return max(decayed, min_lr)

    def step(server_round: int) -> float:
        """Drop the learning rate by decay_rate every step_size rounds."""
        num_drops = _round_index(server_round) // step_size
        return max(initial_lr * decay_rate**num_drops, min_lr)

    builders: dict[str, LRSchedule] = {
        "constant": constant,
        "cosine": cosine,
        "exponential": exponential,
        "step": step,
    }
    return builders[schedule]


def _round_index(server_round: int) -> int:
    """Convert a 1-based server round to a 0-based decay index."""
    return max(server_round, 1) - 1
