"""Parse, validate, and auto-generate partition groups."""

import random

PartitionGroups = dict[int, tuple[int, ...]]

GROUP_MODES = ("manual", "sequential", "random")


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


def _group_sizes(num_partitions: int, num_groups: int) -> list[int]:
    """Split num_partitions into num_groups sizes that differ by at most one."""
    if num_groups <= 0:
        raise ValueError("num_groups must be positive.")
    if num_groups > num_partitions:
        raise ValueError(
            f"num_groups ({num_groups}) cannot exceed num_partitions "
            f"({num_partitions})."
        )

    base_size, remainder = divmod(num_partitions, num_groups)
    return [
        base_size + (1 if group_id <= remainder else 0)
        for group_id in range(1, num_groups + 1)
    ]


def generate_sequential_groups(
    num_partitions: int,
    num_groups: int,
) -> PartitionGroups:
    """Split partitions into contiguous, near-equal-size groups.

    For example, 100 partitions in 4 groups produces
    ``0-24, 25-49, 50-74, 75-99``.
    """
    partition_groups: PartitionGroups = {}
    start = 0
    for group_id, size in enumerate(
        _group_sizes(num_partitions, num_groups), start=1
    ):
        partition_groups[group_id] = tuple(range(start, start + size))
        start += size
    return partition_groups


def generate_random_groups(
    num_partitions: int,
    num_groups: int,
    seed: int | None = None,
) -> PartitionGroups:
    """Randomly split partitions into near-equal-size groups.

    Group sizes match :func:`generate_sequential_groups`; only the
    assignment of partition IDs to groups is randomized.
    """
    sizes = _group_sizes(num_partitions, num_groups)
    shuffled_partitions = list(range(num_partitions))
    random.Random(seed).shuffle(shuffled_partitions)

    partition_groups: PartitionGroups = {}
    start = 0
    for group_id, size in enumerate(sizes, start=1):
        partition_groups[group_id] = tuple(
            sorted(shuffled_partitions[start : start + size])
        )
        start += size
    return partition_groups


def build_partition_groups(
    group_mode: str,
    num_partitions: int,
    num_groups: int,
    manual_specification: str | None = None,
    seed: int | None = None,
) -> PartitionGroups:
    """Build partition groups using the configured mode.

    ``group_mode`` is one of ``"manual"``, ``"sequential"``, or ``"random"``.
    ``manual_specification`` is required (and only used) for ``"manual"``.
    ``seed`` is only used for ``"random"``.
    """
    normalized_mode = group_mode.strip().lower()

    if normalized_mode == "manual":
        if not manual_specification:
            raise ValueError(
                "'partition-groups' must be set when group-mode is 'manual'."
            )
        return parse_partition_groups(
            manual_specification, num_groups, num_partitions
        )
    if normalized_mode == "sequential":
        return generate_sequential_groups(num_partitions, num_groups)
    if normalized_mode == "random":
        return generate_random_groups(num_partitions, num_groups, seed)

    raise ValueError(
        f"Unknown group-mode '{group_mode}'. Expected one of {GROUP_MODES}."
    )