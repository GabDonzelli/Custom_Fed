"""Tests for partition group parsing and validation."""

import unittest

from pytorchexample.strategy.grouping import (
    build_partition_to_group,
    parse_partition_groups,
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


if __name__ == "__main__":
    unittest.main()