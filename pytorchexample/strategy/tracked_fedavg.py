"""Plain FedAvg that additionally tracks per-partition participation."""

from collections.abc import Iterable
from logging import INFO

from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

PARTITION_ID_KEY = "partition_id"


class TrackedFedAvg(FedAvg):
    """Standard FedAvg aggregation, plus a record of which partition replied when.

    Aggregation is identical to :class:`flwr.serverapp.strategy.FedAvg` (no
    grouping) — this only adds bookkeeping of which partition trained in
    which round, for comparison against the grouped strategy.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Counts how many training rounds each partition actually replied to.
        self.partition_participation: dict[int, int] = {}

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Record partition participation, then aggregate like plain FedAvg."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None

        for reply in valid_replies:
            metric_record = next(iter(reply.content.metric_records.values()))
            if PARTITION_ID_KEY not in metric_record:
                raise ValueError(f"Missing '{PARTITION_ID_KEY}' in a training reply.")

            partition_id = int(metric_record.pop(PARTITION_ID_KEY))
            self.partition_participation[partition_id] = (
                self.partition_participation.get(partition_id, 0) + 1
            )

        reply_contents = [msg.content for msg in valid_replies]
        arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return arrays, metrics

    def log_participation_summary(self, num_partitions: int, num_rounds: int) -> None:
        """Log how many training rounds each partition replied to."""
        for partition_id in range(num_partitions):
            count = self.partition_participation.get(partition_id, 0)
            log(
                INFO,
                "partition %d: participated in %d/%d training rounds",
                partition_id,
                count,
                num_rounds,
            )
