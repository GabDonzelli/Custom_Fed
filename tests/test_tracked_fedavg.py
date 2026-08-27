"""Tests for TrackedFedAvg: plain FedAvg aggregation plus participation counts."""

import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MessageType, MetricRecord, RecordDict

from pytorchexample.strategy.tracked_fedavg import TrackedFedAvg


def make_reply(node_id: int, partition_id: int, model_value: float, num_examples: int) -> Message:
    """Build a synthetic training reply as if it came from one client."""
    request = Message(
        content=RecordDict(), message_type=MessageType.TRAIN, dst_node_id=node_id
    )
    content = RecordDict(
        {
            "arrays": ArrayRecord(
                {"w": Array(np.array([model_value], dtype=np.float32))}
            ),
            "metrics": MetricRecord(
                {"num-examples": num_examples, "partition_id": partition_id}
            ),
        }
    )
    return Message(content=content, reply_to=request)


class TrackedFedAvgAggregationTest(unittest.TestCase):
    """Verify aggregation still matches plain weighted FedAvg."""

    def test_aggregate_train_matches_plain_weighted_average(self) -> None:
        """Aggregation ignores groups entirely — a single weighted average."""
        strategy = TrackedFedAvg(fraction_train=1.0, min_train_nodes=1, min_available_nodes=1)

        arrays, metrics = strategy.aggregate_train(
            server_round=1,
            replies=[
                make_reply(node_id=10, partition_id=0, model_value=1.0, num_examples=10),
                make_reply(node_id=11, partition_id=1, model_value=2.0, num_examples=20),
                make_reply(node_id=12, partition_id=2, model_value=4.0, num_examples=30),
            ],
        )

        expected = (1 * 10 + 2 * 20 + 4 * 30) / 60
        self.assertAlmostEqual(float(arrays["w"].numpy()[0]), expected, places=6)
        # partition_id must not leak into the aggregated metrics.
        self.assertNotIn("partition_id", metrics)

    def test_no_valid_replies_returns_none(self) -> None:
        """An empty reply list aggregates to nothing, same as plain FedAvg."""
        strategy = TrackedFedAvg(fraction_train=1.0, min_train_nodes=1, min_available_nodes=1)

        arrays, metrics = strategy.aggregate_train(server_round=1, replies=[])
        self.assertIsNone(arrays)
        self.assertIsNone(metrics)


class TrackedFedAvgParticipationTest(unittest.TestCase):
    """Verify per-partition participation counting across rounds."""

    def test_counts_only_partitions_that_actually_replied(self) -> None:
        """A partition that never replies stays out of the summary at zero."""
        strategy = TrackedFedAvg(fraction_train=1.0, min_train_nodes=1, min_available_nodes=1)

        # Round 1: all four partitions reply.
        strategy.aggregate_train(
            server_round=1,
            replies=[
                make_reply(node_id=10, partition_id=0, model_value=1.0, num_examples=5),
                make_reply(node_id=11, partition_id=1, model_value=1.0, num_examples=5),
                make_reply(node_id=12, partition_id=2, model_value=1.0, num_examples=5),
                make_reply(node_id=13, partition_id=3, model_value=1.0, num_examples=5),
            ],
        )
        # Round 2: only partitions 0 and 2 reply (e.g. fraction_train < 1.0).
        strategy.aggregate_train(
            server_round=2,
            replies=[
                make_reply(node_id=10, partition_id=0, model_value=1.0, num_examples=5),
                make_reply(node_id=12, partition_id=2, model_value=1.0, num_examples=5),
            ],
        )

        self.assertEqual(strategy.partition_participation[0], 2)
        self.assertEqual(strategy.partition_participation[1], 1)
        self.assertEqual(strategy.partition_participation[2], 2)
        self.assertEqual(strategy.partition_participation[3], 1)

    def test_log_participation_summary_covers_every_partition(self) -> None:
        """A partition with zero replies still appears in the summary."""
        strategy = TrackedFedAvg(fraction_train=1.0, min_train_nodes=1, min_available_nodes=1)
        strategy.aggregate_train(
            server_round=1,
            replies=[make_reply(node_id=10, partition_id=0, model_value=1.0, num_examples=5)],
        )

        summarized = {
            partition_id: strategy.partition_participation.get(partition_id, 0)
            for partition_id in range(4)
        }
        self.assertEqual(summarized, {0: 1, 1: 0, 2: 0, 3: 0})
        # Exercise the logging path itself for regressions (no exception).
        strategy.log_participation_summary(num_partitions=4, num_rounds=1)


if __name__ == "__main__":
    unittest.main()
