"""Tests for partition group parsing, validation, and auto-generation."""

import unittest

from pytorchexample.strategy.grouping import (
    build_partition_groups,
    build_partition_to_group,
    generate_random_groups,
    generate_sequential_groups,
    parse_partition_groups,
    validate_partition_groups,
)


class ParsePartitionGroupsTest(unittest.TestCase):
    """Verify valid and invalid group configurations."""

    def test_parse_four_groups(self) -> None:
        """Parse all ten partitions into four groups."""
        groups = parse_partition_groups(
            "0,1,2|3,4,5|6,7|8,9",
            num_groups=4,
            num_partitions=10,
        )

        self.assertEqual(groups[1], (0, 1, 2))
        self.assertEqual(groups[4], (8, 9))
        self.assertEqual(build_partition_to_group(groups)[6], 3)

    def test_reject_duplicate_partition(self) -> None:
        """Reject a partition assigned to more than one group."""
        with self.assertRaisesRegex(ValueError, "more than one group"):
            parse_partition_groups(
                "0,1,2|2,3,4|5,6|7,8,9",
                num_groups=4,
                num_partitions=10,
            )

    def test_reject_missing_partition(self) -> None:
        """Reject a configuration that omits an expected partition."""
        with self.assertRaisesRegex(ValueError, "Missing"):
            parse_partition_groups(
                "0,1,2|3,4|5,6|7,8",
                num_groups=4,
                num_partitions=10,
            )

    def test_reject_wrong_number_of_groups(self) -> None:
        """Reject a group count that differs from the run configuration."""
        with self.assertRaisesRegex(ValueError, "Expected 4 groups"):
            parse_partition_groups(
                "0,1,2|3,4,5|6,7,8,9",
                num_groups=4,
                num_partitions=10,
            )


class GenerateSequentialGroupsTest(unittest.TestCase):
    """Verify contiguous, near-equal-size group generation."""

    def test_even_split(self) -> None:
        """100 partitions in 4 groups split into exact quarters."""
        groups = generate_sequential_groups(num_partitions=100, num_groups=4)

        self.assertEqual(groups[1], tuple(range(0, 25)))
        self.assertEqual(groups[2], tuple(range(25, 50)))
        self.assertEqual(groups[3], tuple(range(50, 75)))
        self.assertEqual(groups[4], tuple(range(75, 100)))
        validate_partition_groups(groups, num_partitions=100)

    def test_uneven_split_distributes_remainder(self) -> None:
        """10 partitions in 4 groups gives sizes 3,3,2,2."""
        groups = generate_sequential_groups(num_partitions=10, num_groups=4)

        self.assertEqual([len(ids) for ids in groups.values()], [3, 3, 2, 2])
        validate_partition_groups(groups, num_partitions=10)

    def test_rejects_more_groups_than_partitions(self) -> None:
        """A group count above the partition count is invalid."""
        with self.assertRaises(ValueError):
            generate_sequential_groups(num_partitions=3, num_groups=4)


class GenerateRandomGroupsTest(unittest.TestCase):
    """Verify randomized group generation."""

    def test_covers_every_partition_exactly_once(self) -> None:
        """A random split still forms a valid partition of all IDs."""
        groups = generate_random_groups(num_partitions=100, num_groups=4, seed=0)

        validate_partition_groups(groups, num_partitions=100)
        self.assertEqual([len(ids) for ids in groups.values()], [25, 25, 25, 25])

    def test_seed_is_reproducible(self) -> None:
        """The same seed produces the same group assignment."""
        first = generate_random_groups(num_partitions=20, num_groups=4, seed=42)
        second = generate_random_groups(num_partitions=20, num_groups=4, seed=42)

        self.assertEqual(first, second)

    def test_differs_from_sequential_assignment(self) -> None:
        """A random split (almost certainly) reorders membership."""
        sequential = generate_sequential_groups(num_partitions=20, num_groups=4)
        random_groups = generate_random_groups(num_partitions=20, num_groups=4, seed=1)

        self.assertNotEqual(sequential, random_groups)


class BuildPartitionGroupsTest(unittest.TestCase):
    """Verify the group-mode dispatcher."""

    def test_manual_mode_uses_specification(self) -> None:
        """Manual mode parses the given specification string."""
        groups = build_partition_groups(
            group_mode="manual",
            num_partitions=10,
            num_groups=4,
            manual_specification="0,1,2|3,4,5|6,7|8,9",
        )
        self.assertEqual(groups[1], (0, 1, 2))

    def test_manual_mode_requires_specification(self) -> None:
        """Manual mode without a specification raises a clear error."""
        with self.assertRaisesRegex(ValueError, "partition-groups"):
            build_partition_groups(
                group_mode="manual",
                num_partitions=10,
                num_groups=4,
            )

    def test_sequential_mode(self) -> None:
        """Sequential mode delegates to generate_sequential_groups."""
        groups = build_partition_groups(
            group_mode="sequential",
            num_partitions=100,
            num_groups=4,
        )
        self.assertEqual(groups[1], tuple(range(0, 25)))

    def test_random_mode(self) -> None:
        """Random mode delegates to generate_random_groups."""
        groups = build_partition_groups(
            group_mode="random",
            num_partitions=20,
            num_groups=4,
            seed=7,
        )
        validate_partition_groups(groups, num_partitions=20)

    def test_unknown_mode_rejected(self) -> None:
        """An unsupported group-mode raises a clear error."""
        with self.assertRaisesRegex(ValueError, "Unknown group-mode"):
            build_partition_groups(
                group_mode="chaotic",
                num_partitions=10,
                num_groups=4,
            )


if __name__ == "__main__":
    unittest.main()