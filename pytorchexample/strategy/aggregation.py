"""Pure aggregation helpers for the two-level FedAvg calculation."""

from collections.abc import Callable
from dataclasses import dataclass

from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords


@dataclass(frozen=True)
class GroupAggregation:
    """Store one temporary group model and its aggregation metadata."""

    group_id: int
    arrays: ArrayRecord
    metrics: MetricRecord
    num_examples: float
    partition_ids: tuple[int, ...]


def _get_record_weight(record: RecordDict, weighted_by_key: str) -> float:
    """Read the aggregation weight from the only MetricRecord in a reply."""
    metric_record = next(iter(record.metric_records.values()))
    return float(metric_record[weighted_by_key])


def aggregate_records_in_group(
    group_id: int,
    records: list[RecordDict],
    partition_ids: list[int],
    weighted_by_key: str,
    metrics_aggregation_fn: Callable[[list[RecordDict], str], MetricRecord],
) -> GroupAggregation:
    """Aggregate client models and metrics inside one group."""
    if not records:
        raise ValueError(f"Group {group_id} has no successful client replies.")

    num_examples = sum(
        _get_record_weight(record, weighted_by_key) for record in records
    )
    if num_examples <= 0:
        raise ValueError(f"Group {group_id} has a non-positive aggregation weight.")

    return GroupAggregation(
        group_id=group_id,
        arrays=aggregate_arrayrecords(records, weighted_by_key),
        metrics=metrics_aggregation_fn(records, weighted_by_key),
        num_examples=num_examples,
        partition_ids=tuple(sorted(partition_ids)),
    )


def aggregate_group_models(
    group_aggregations: list[GroupAggregation],
    weighted_by_key: str,
    arrayrecord_key: str,
) -> ArrayRecord:
    """Aggregate group models proportionally to each group's examples."""
    group_records = [
        RecordDict(
            {
                arrayrecord_key: group.arrays,
                "group-weight": MetricRecord({weighted_by_key: group.num_examples}),
            }
        )
        for group in group_aggregations
    ]
    return aggregate_arrayrecords(group_records, weighted_by_key)