"""Integration tests for aggregation with Flower record types."""

import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from pytorchexample.strategy.aggregation import (
    aggregate_group_models,
    aggregate_records_in_group,
)


def make_record(model_value: float, num_examples: int) -> RecordDict:
    """Create one minimal client reply for aggregation tests."""
    arrays = ArrayRecord(
        {
            "weight": Array(
                np.array([model_value], dtype=np.float32)
            )
        }
    )
    metrics = MetricRecord(
        {
            "num-examples": num_examples,
            "train_loss": model_value,
        }
    )
    return RecordDict({"arrays": arrays, "metrics": metrics})


class FlowerAggregationTest(unittest.TestCase):
    """Verify the two aggregation levels with real Flower records."""

    def test_group_models_use_group_example_totals(self) -> None:
        """Match the direct client-level weighted average."""
        client_records = [
            make_record(1.0, 10),
            make_record(2.0, 20),
            make_record(4.0, 30),
            make_record(8.0, 15),
            make_record(16.0, 25),
        ]

        record_groups = [
            client_records[0:2],
            client_records[2:3],
            client_records[3:4],
            client_records[4:5],
        ]

        group_aggregations = [
            aggregate_records_in_group(
                group_id=group_id,
                records=records,
                partition_ids=[group_id - 1],
                weighted_by_key="num-examples",
                metrics_aggregation_fn=aggregate_metricrecords,
            )
            for group_id, records in enumerate(
                record_groups,
                start=1,
            )
        ]

        grouped_arrays = aggregate_group_models(
            group_aggregations=group_aggregations,
            weighted_by_key="num-examples",
            arrayrecord_key="arrays",
        )

        expected = (
            1 * 10
            + 2 * 20
            + 4 * 30
            + 8 * 15
            + 16 * 25
        ) / 100

        actual = float(
            grouped_arrays["weight"].numpy()[0]
        )

        self.assertAlmostEqual(
            actual,
            expected,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()