"""Flower strategy that performs proportional two-level FedAvg."""

from collections.abc import Iterable
from logging import INFO, WARNING

from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp.strategy import FedAvg

from pytorchexample.strategy.aggregation import (
    GroupAggregation,
    aggregate_group_models,
    aggregate_records_in_group,
)
from pytorchexample.strategy.grouping import (
    PartitionGroups,
    build_partition_to_group,
)

PARTITION_ID_KEY = "partition_id"


class GroupedFedAvg(FedAvg):
    """Aggregate client models inside groups and then across groups."""

    def __init__(
        self,
        partition_groups: PartitionGroups,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.partition_groups = partition_groups
        self.partition_to_group = build_partition_to_group(partition_groups)

        # This mapping is learned from real replies, never from node ordering.
        self.partition_to_node: dict[int, int] = {}

        # The history contains small summaries, not model parameters.
        self.group_history: dict[int, dict[int, dict[str, object]]] = {}

        # Counts how many training rounds each partition actually replied to.
        self.partition_participation: dict[int, int] = {}

        # How many clients contributed to the most recent training round. Read
        # by the ServerApp so the results CSV records it: a run where every
        # client fails otherwise looks identical to a run that legitimately
        # did not improve.
        self.last_train_client_count: int = 0

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Perform client-to-group and group-to-global aggregation."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            # Returning None keeps the previous global arrays, so the run walks
            # on and the centralized metrics repeat verbatim every round. That
            # reads exactly like "the model did not improve" while actually
            # meaning "nothing was ever trained" -- so say so loudly.
            self.last_train_client_count = 0
            log(
                WARNING,
                "round %d trained NOTHING: every client reply was invalid or "
                "missing, so the global model is unchanged. A common cause is "
                "'num-supernodes' not matching 'num-partitions', which makes "
                "every client raise in _read_partition_config.",
                server_round,
            )
            return None, None

        records_by_group = {group_id: [] for group_id in self.partition_groups}
        partitions_by_group = {group_id: [] for group_id in self.partition_groups}
        all_records = []
        received_partitions: set[int] = set()

        for reply in valid_replies:
            record = reply.content
            metric_record = next(iter(record.metric_records.values()))

            if PARTITION_ID_KEY not in metric_record:
                raise ValueError(f"Missing '{PARTITION_ID_KEY}' in a training reply.")

            partition_id = int(metric_record.pop(PARTITION_ID_KEY))
            if partition_id in received_partitions:
                raise ValueError(
                    f"Received more than one reply for partition {partition_id}."
                )
            if partition_id not in self.partition_to_group:
                raise ValueError(
                    f"Partition {partition_id} does not belong to a configured group."
                )

            received_partitions.add(partition_id)
            node_id = reply.metadata.src_node_id
            self.partition_to_node[partition_id] = node_id
            self.partition_participation[partition_id] = (
                self.partition_participation.get(partition_id, 0) + 1
            )

            group_id = self.partition_to_group[partition_id]
            records_by_group[group_id].append(record)
            partitions_by_group[group_id].append(partition_id)
            all_records.append(record)

        group_aggregations: list[GroupAggregation] = []
        for group_id in sorted(self.partition_groups):
            group = aggregate_records_in_group(
                group_id=group_id,
                records=records_by_group[group_id],
                partition_ids=partitions_by_group[group_id],
                weighted_by_key=self.weighted_by_key,
                metrics_aggregation_fn=self.train_metrics_aggr_fn,
            )
            group_aggregations.append(group)
            log(
                INFO,
                "round %d, group %d: examples=%d",
                server_round,
                group.group_id,
                int(group.num_examples),
            )

        global_arrays = aggregate_group_models(
            group_aggregations=group_aggregations,
            weighted_by_key=self.weighted_by_key,
            arrayrecord_key=self.arrayrecord_key,
        )
        global_metrics = self.train_metrics_aggr_fn(
            all_records,
            self.weighted_by_key,
        )
        self.last_train_client_count = len(received_partitions)

        self.group_history[server_round] = {
            group.group_id: {
                "partition_ids": list(group.partition_ids),
                "num_clients": len(group.partition_ids),
                "num_examples": group.num_examples,
                "metrics": dict(group.metrics),
            }
            for group in group_aggregations
        }

        log(
            INFO,
            "round %d: aggregated %d proportional group models",
            server_round,
            len(group_aggregations),
        )
        return global_arrays, global_metrics

    def log_participation_summary(self, num_rounds: int) -> None:
        """Log how many training rounds each configured partition replied to."""
        for partition_id in sorted(self.partition_to_group):
            count = self.partition_participation.get(partition_id, 0)
            log(
                INFO,
                "partition %d: participated in %d/%d training rounds",
                partition_id,
                count,
                num_rounds,
            )