"""Tests for GroupedFedAvg orchestration and participation tracking."""

import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MessageType, MetricRecord, RecordDict

from pytorchexample.strategy.grouped_fedavg import GroupedFedAvg


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

        One reply per group here, so partitions 1 and 3 end the round with zero
        participation while both groups still contribute.
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


class GroupedFedAvgPartialParticipationTest(unittest.TestCase):
    """Verify what happens when a whole group goes unsampled in a round.

    This is not an edge case once fraction_train < 1.0. The server samples
    clients without regard to grouping, so with 10 partitions in 4 groups and 7
    sampled per round, some group receives no client in roughly 15% of rounds --
    which means it happens in nearly every 20-round run.
    """

    def _make_strategy(self) -> GroupedFedAvg:
        return GroupedFedAvg(
            partition_groups={1: (0, 1), 2: (2, 3)},
            fraction_train=0.5,
            min_train_nodes=1,
            min_available_nodes=1,
        )

    def test_an_unsampled_group_is_skipped_not_fatal(self) -> None:
        """The round still aggregates, using only the groups that replied."""
        strategy = self._make_strategy()

        arrays, metrics = strategy.aggregate_train(
            server_round=1,
            replies=[
                make_reply(node_id=10, partition_id=0, num_examples=5),
                make_reply(node_id=11, partition_id=1, num_examples=5),
            ],
        )

        self.assertIsNotNone(arrays)
        self.assertIsNotNone(metrics)
        self.assertEqual(strategy.last_train_client_count, 2)
        # Only group 1 contributed, so only group 1 appears in the history.
        self.assertEqual(list(strategy.group_history[1]), [1])

    def test_skipping_a_group_does_not_change_the_global_model(self) -> None:
        """A group with no data must not be invented, only left out.

        Averaging group 1's model with a phantom group 2 would drag the global
        model toward whatever the phantom held. The result of a round where only
        group 1 replied must therefore equal group 1's own aggregate.
        """
        strategy = self._make_strategy()

        arrays, _ = strategy.aggregate_train(
            server_round=1,
            replies=[
                make_reply(node_id=10, partition_id=0, num_examples=5),
                make_reply(node_id=11, partition_id=1, num_examples=5),
            ],
        )

        # Every synthetic reply carries w = 1.0, so any weighted average of the
        # replies that actually arrived is exactly 1.0 -- and a phantom group
        # contributing zeros would show up here as something smaller.
        weights = arrays["w"].numpy()
        self.assertAlmostEqual(float(weights[0]), 1.0, places=6)

    def test_a_round_where_no_group_replies_keeps_the_previous_model(self) -> None:
        """Zero valid replies is still the "nothing trained" case, not a crash."""
        strategy = self._make_strategy()

        arrays, metrics = strategy.aggregate_train(server_round=1, replies=[])

        self.assertIsNone(arrays)
        self.assertIsNone(metrics)
        self.assertEqual(strategy.last_train_client_count, 0)

if __name__ == "__main__":
    unittest.main()
