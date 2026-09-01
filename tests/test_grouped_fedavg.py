"""Tests for GroupedFedAvg orchestration and participation tracking."""

import unittest

import numpy as np
from flwr.app import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)

from pytorchexample.strategy.grouped_fedavg import GroupedFedAvg
from pytorchexample.strategy.lr_schedule import build_lr_schedule


class FakeGrid:
    """Minimal Grid stand-in exposing only the node ids FedAvg samples from."""

    def __init__(self, node_ids: list[int]) -> None:
        self._node_ids = node_ids

    def get_node_ids(self) -> list[int]:
        """Return the connected node ids."""
        return self._node_ids


def make_reply(node_id: int, partition_id: int, num_examples: int) -> Message:
    """Build a synthetic training reply as if it came from one client."""
    request = Message(
        content=RecordDict(), message_type=MessageType.TRAIN, dst_node_id=node_id
    )
    content = RecordDict(
        {
            "arrays": ArrayRecord({"w": Array(np.array([1.0], dtype=np.float32))}),
            "metrics": MetricRecord(
                {"num-examples": num_examples, "partition_id": partition_id}
            ),
        }
    )
    return Message(content=content, reply_to=request)


class GroupedFedAvgParticipationTest(unittest.TestCase):
    """Verify per-partition participation counting across rounds."""

    def _make_strategy(self) -> GroupedFedAvg:
        return GroupedFedAvg(
            partition_groups={1: (0, 1), 2: (2, 3)},
            fraction_train=1.0,
            min_train_nodes=1,
            min_available_nodes=1,
        )

    def test_counts_only_partitions_that_actually_replied(self) -> None:
        """A partition that never replies stays out of the summary at zero."""
        strategy = self._make_strategy()

        # Round 1: all four partitions reply.
        strategy.aggregate_train(
            server_round=1,
            replies=[
                make_reply(node_id=10, partition_id=0, num_examples=5),
                make_reply(node_id=11, partition_id=1, num_examples=5),
                make_reply(node_id=12, partition_id=2, num_examples=5),
                make_reply(node_id=13, partition_id=3, num_examples=5),
            ],
        )
        # Round 2: only partitions 0 and 2 reply (e.g. fraction_train < 1.0).
        strategy.aggregate_train(
            server_round=2,
            replies=[
                make_reply(node_id=10, partition_id=0, num_examples=5),
                make_reply(node_id=12, partition_id=2, num_examples=5),
            ],
        )

        self.assertEqual(strategy.partition_participation[0], 2)
        self.assertEqual(strategy.partition_participation[1], 1)
        self.assertEqual(strategy.partition_participation[2], 2)
        self.assertEqual(strategy.partition_participation[3], 1)

    def test_log_participation_summary_covers_every_configured_partition(
        self,
    ) -> None:
        """A partition with zero replies still appears in the summary.

        Each group needs at least one reply per round (aggregate_train raises
        otherwise), so this covers both groups (partitions 0 and 2) while
        leaving partitions 1 and 3 with zero participation.
        """
        strategy = self._make_strategy()
        strategy.aggregate_train(
            server_round=1,
            replies=[
                make_reply(node_id=10, partition_id=0, num_examples=5),
                make_reply(node_id=12, partition_id=2, num_examples=5),
            ],
        )

        # partition_to_group holds every configured partition, so a summary
        # built from it must not silently drop partitions that never replied.
        summarized = {
            partition_id: strategy.partition_participation.get(partition_id, 0)
            for partition_id in sorted(strategy.partition_to_group)
        }
        self.assertEqual(summarized, {0: 1, 1: 0, 2: 1, 3: 0})
        # Exercise the logging path itself for regressions (no exception).
        strategy.log_participation_summary(num_rounds=1)


class GroupedFedAvgLearningRateTest(unittest.TestCase):
    """Verify the scheduled learning rate reaches the outgoing train messages."""

    def _configure_round(
        self, strategy: GroupedFedAvg, server_round: int
    ) -> list[Message]:
        """Run one configure_train and return the messages sent to clients."""
        return list(
            strategy.configure_train(
                server_round=server_round,
                arrays=ArrayRecord({"w": Array(np.array([1.0], dtype=np.float32))}),
                config=ConfigRecord({"lr": 0.1}),
                grid=FakeGrid([10, 11, 12, 13]),
            )
        )

    def test_broadcasts_the_scheduled_rate_for_each_round(self) -> None:
        """Round 1 gets the initial rate and the last round gets min_lr."""
        strategy = GroupedFedAvg(
            partition_groups={1: (0, 1), 2: (2, 3)},
            lr_schedule=build_lr_schedule(
                schedule="cosine", initial_lr=0.1, num_rounds=100, min_lr=0.001
            ),
            fraction_train=1.0,
            min_train_nodes=1,
            min_available_nodes=1,
        )

        first_round = self._configure_round(strategy, server_round=1)
        self.assertTrue(first_round)
        for message in first_round:
            self.assertAlmostEqual(message.content["config"]["lr"], 0.1)

        last_round = self._configure_round(strategy, server_round=100)
        self.assertTrue(last_round)
        for message in last_round:
            self.assertAlmostEqual(message.content["config"]["lr"], 0.001)

    def test_without_a_schedule_the_config_rate_is_untouched(self) -> None:
        """Omitting the schedule must keep the constant-LR behaviour."""
        strategy = GroupedFedAvg(
            partition_groups={1: (0, 1), 2: (2, 3)},
            fraction_train=1.0,
            min_train_nodes=1,
            min_available_nodes=1,
        )

        for server_round in (1, 50, 100):
            for message in self._configure_round(strategy, server_round=server_round):
                self.assertAlmostEqual(message.content["config"]["lr"], 0.1)


if __name__ == "__main__":
    unittest.main()
