"""Verify the mathematical equivalence of proportional two-level FedAvg."""

import unittest


def weighted_average(values: list[float], weights: list[float]) -> float:
    """Return a small scalar weighted average used by this unit test."""
    total_weight = sum(weights)
    return sum(
        value * weight / total_weight
        for value, weight in zip(values, weights, strict=True)
    )


class ProportionalAggregationTest(unittest.TestCase):
    """Compare direct FedAvg with proportional group aggregation."""

    def test_two_level_average_equals_direct_average(self) -> None:
        """Use group example totals as the second-level weights."""
        client_models = [1.0, 2.0, 4.0, 8.0, 16.0]
        client_examples = [10.0, 20.0, 30.0, 15.0, 25.0]
        group_indices = [(0, 1), (2,), (3,), (4,)]

        group_models = []
        group_examples = []

        for indices in group_indices:
            values = [client_models[index] for index in indices]
            weights = [client_examples[index] for index in indices]

            group_models.append(weighted_average(values, weights))
            group_examples.append(sum(weights))

        direct_result = weighted_average(client_models, client_examples)
        grouped_result = weighted_average(group_models, group_examples)

        self.assertAlmostEqual(grouped_result, direct_result)


if __name__ == "__main__":
    unittest.main()