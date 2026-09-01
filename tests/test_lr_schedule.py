"""Verify the round-based learning-rate schedules."""

import unittest

from pytorchexample.strategy.lr_schedule import LR_SCHEDULES, build_lr_schedule


class LRScheduleTest(unittest.TestCase):
    """Check the decay shape and boundaries of each schedule."""

    def test_every_schedule_starts_at_the_initial_rate(self) -> None:
        """Round 1 must always train with the configured learning rate."""
        for schedule in LR_SCHEDULES:
            with self.subTest(schedule=schedule):
                lr_schedule = build_lr_schedule(
                    schedule=schedule,
                    initial_lr=0.1,
                    num_rounds=200,
                    min_lr=0.001,
                    decay_rate=0.9,
                    step_size=50,
                )
                self.assertAlmostEqual(lr_schedule(1), 0.1)

    def test_round_zero_matches_round_one(self) -> None:
        """The pre-training evaluation round must not extrapolate backwards."""
        for schedule in LR_SCHEDULES:
            with self.subTest(schedule=schedule):
                lr_schedule = build_lr_schedule(
                    schedule=schedule, initial_lr=0.1, num_rounds=200
                )
                self.assertAlmostEqual(lr_schedule(0), lr_schedule(1))

    def test_constant_never_decays(self) -> None:
        """The no-decay baseline keeps one rate for the whole run."""
        lr_schedule = build_lr_schedule(
            schedule="constant", initial_lr=0.1, num_rounds=100
        )
        self.assertAlmostEqual(lr_schedule(1), 0.1)
        self.assertAlmostEqual(lr_schedule(100), 0.1)

    def test_cosine_reaches_min_lr_on_the_final_round(self) -> None:
        """Cosine annealing must land exactly on min_lr, not above it."""
        lr_schedule = build_lr_schedule(
            schedule="cosine", initial_lr=0.1, num_rounds=150, min_lr=0.001
        )
        self.assertAlmostEqual(lr_schedule(150), 0.001)

    def test_cosine_is_monotonically_decreasing(self) -> None:
        """The rate must never rise between consecutive rounds."""
        lr_schedule = build_lr_schedule(
            schedule="cosine", initial_lr=0.1, num_rounds=150, min_lr=0.001
        )
        rates = [lr_schedule(server_round) for server_round in range(1, 151)]
        for earlier, later in zip(rates[:-1], rates[1:], strict=True):
            self.assertLessEqual(later, earlier)

    def test_cosine_with_a_single_round_returns_the_initial_rate(self) -> None:
        """A one-round run has no interval to anneal over."""
        lr_schedule = build_lr_schedule(
            schedule="cosine", initial_lr=0.1, num_rounds=1, min_lr=0.001
        )
        self.assertAlmostEqual(lr_schedule(1), 0.1)

    def test_exponential_decays_once_per_round(self) -> None:
        """Round n must be initial_lr * decay_rate ** (n - 1)."""
        lr_schedule = build_lr_schedule(
            schedule="exponential", initial_lr=0.1, num_rounds=100, decay_rate=0.9
        )
        self.assertAlmostEqual(lr_schedule(2), 0.1 * 0.9)
        self.assertAlmostEqual(lr_schedule(4), 0.1 * 0.9**3)

    def test_exponential_is_floored_at_min_lr(self) -> None:
        """Aggressive decay must not drive the rate below the floor."""
        lr_schedule = build_lr_schedule(
            schedule="exponential",
            initial_lr=0.1,
            num_rounds=1000,
            min_lr=0.01,
            decay_rate=0.5,
        )
        self.assertAlmostEqual(lr_schedule(1000), 0.01)

    def test_step_holds_the_rate_inside_a_step(self) -> None:
        """The rate drops only when a step boundary is crossed."""
        lr_schedule = build_lr_schedule(
            schedule="step",
            initial_lr=0.1,
            num_rounds=100,
            decay_rate=0.5,
            step_size=10,
        )
        self.assertAlmostEqual(lr_schedule(10), 0.1)
        self.assertAlmostEqual(lr_schedule(11), 0.05)
        self.assertAlmostEqual(lr_schedule(20), 0.05)
        self.assertAlmostEqual(lr_schedule(21), 0.025)

    def test_unknown_schedule_is_rejected(self) -> None:
        """A typo in the config must fail loudly, not silently disable decay."""
        with self.assertRaises(ValueError):
            build_lr_schedule(schedule="cosin", initial_lr=0.1, num_rounds=10)

    def test_min_lr_above_initial_lr_is_rejected(self) -> None:
        """An inverted range would make the schedule increase the rate."""
        with self.assertRaises(ValueError):
            build_lr_schedule(
                schedule="cosine", initial_lr=0.01, num_rounds=10, min_lr=0.1
            )

    def test_out_of_range_decay_rate_is_rejected(self) -> None:
        """A decay rate above 1 would grow the learning rate every round."""
        with self.assertRaises(ValueError):
            build_lr_schedule(
                schedule="exponential", initial_lr=0.1, num_rounds=10, decay_rate=1.5
            )
        with self.assertRaises(ValueError):
            build_lr_schedule(
                schedule="exponential", initial_lr=0.1, num_rounds=10, decay_rate=0.0
            )


if __name__ == "__main__":
    unittest.main()
