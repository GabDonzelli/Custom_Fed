"""Parse and validate partition groups."""

PartitionGroups = dict[int, tuple[int, ...]]


def parse_partition_groups(
    group_specification: str,
    num_groups: int,
    num_partitions: int,
) -> PartitionGroups:
    """Parse a compact group specification into validated partition groups.

    Groups are separated by ``|`` and partition IDs by ``,``. For example,
    ``0,1,2|3,4,5|6,7|8,9`` defines four groups for ten partitions.
    """
    raw_groups = [item.strip() for item in group_specification.split("|")]
    if len(raw_groups) != num_groups:
        raise ValueError(
            f"Expected {num_groups} groups, but received {len(raw_groups)}."
        )

    partition_groups: PartitionGroups = {}
    for group_id, raw_group in enumerate(raw_groups, start=1):
        if not raw_group:
            raise ValueError(f"Group {group_id} cannot be empty.")

        try:
            partition_ids = tuple(int(item.strip()) for item in raw_group.split(","))
        except ValueError as exc:
            raise ValueError(
                f"Group {group_id} contains a non-integer partition ID."
            ) from exc

        if len(set(partition_ids)) != len(partition_ids):
            raise ValueError(f"Group {group_id} contains duplicate partition IDs.")
        partition_groups[group_id] = partition_ids

    validate_partition_groups(partition_groups, num_partitions)
    return partition_groups


def validate_partition_groups(
    partition_groups: PartitionGroups,
    num_partitions: int,
) -> None:
    """Ensure every expected partition appears in exactly one group."""
    configured_partitions = [
        partition_id
        for partition_ids in partition_groups.values()
        for partition_id in partition_ids
    ]
    configured_set = set(configured_partitions)
    expected_set = set(range(num_partitions))

    if len(configured_partitions) != len(configured_set):
        raise ValueError("A partition ID cannot belong to more than one group.")

    missing = sorted(expected_set - configured_set)
    unexpected = sorted(configured_set - expected_set)
    if missing or unexpected:
        raise ValueError(
            "Partition groups do not match the expected partitions. "
            f"Missing: {missing}; unexpected: {unexpected}."
        )


def build_partition_to_group(
    partition_groups: PartitionGroups,
) -> dict[int, int]:
    """Create a direct lookup from partition ID to group ID."""
    return {
        partition_id: group_id
        for group_id, partition_ids in partition_groups.items()
        for partition_id in partition_ids
    }