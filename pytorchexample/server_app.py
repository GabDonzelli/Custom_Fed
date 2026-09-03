"""ServerApp entry point for proportional grouped FedAvg."""
from logging import INFO

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from pytorchexample.strategy.grouped_fedavg import GroupedFedAvg
from pytorchexample.strategy.grouping import build_partition_groups
from pytorchexample.tasks.registry import get_task
from pytorchexample.results_logger import ResultsLogger

app = ServerApp()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run federated training with four proportional client groups."""
    task_name = str(context.run_config["task-name"])
    num_partitions = int(context.run_config["num-partitions"])
    num_groups = int(context.run_config["num-groups"])
    batch_size = int(context.run_config["batch-size"])

    group_mode = str(context.run_config.get("group-mode", "sequential"))
    manual_specification = context.run_config.get("partition-groups")
    group_seed = context.run_config.get("group-seed")

    partition_groups = build_partition_groups(
        group_mode=group_mode,
        num_partitions=num_partitions,
        num_groups=num_groups,
        manual_specification=(
            str(manual_specification) if manual_specification is not None else None
        ),
        seed=int(group_seed) if group_seed is not None else None,
    )
    log(INFO, "partition groups built using group-mode=%r", group_mode)
    for group_id, partition_ids in partition_groups.items():
        log(
            INFO,
            "configured group %d with partitions %s",
            group_id,
            list(partition_ids),
        )

    task = get_task(task_name)
    global_model = task.create_model()
    initial_arrays = ArrayRecord(task.get_federated_arrays(global_model))
    centralized_testloader = task.load_centralized_data(
        num_partitions=num_partitions,
        batch_size=batch_size,
    )

    strategy = GroupedFedAvg(
        partition_groups=partition_groups,
        fraction_train=float(context.run_config.get("fraction-train", 1.0)),
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_train_nodes=int(context.run_config.get("min-train-nodes", 2)),
        # Flower defaults this to 2 as well. Left unset, a run with a single
        # client trains fine and then hangs forever on "Waiting for nodes to
        # connect: 1 connected (minimum required: 2)" in the evaluate phase.
        min_evaluate_nodes=int(context.run_config.get("min-evaluate-nodes", 2)),
        min_available_nodes=num_partitions,
        weighted_by_key="num-examples",
    )

    results_logger = ResultsLogger()

    def global_evaluate(
        server_round: int,
        arrays: ArrayRecord,
    ) -> MetricRecord:
        """Evaluate the current global model on centralized test data."""
        model = task.create_model()
        task.load_federated_arrays(model, arrays.to_torch_state_dict())
        evaluation_metrics = task.evaluate(model, centralized_testloader, DEVICE)

        accuracy = evaluation_metrics.get("accuracy", 0.0)
        loss = evaluation_metrics.get("loss", 0.0)
        results_logger.log_round(
            round_num=server_round,
            accuracy=accuracy,
            loss=loss,
            num_clients=strategy.last_train_client_count,
        )

        return MetricRecord(evaluation_metrics)

    num_rounds = int(context.run_config["num-server-rounds"])
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": float(context.run_config["learning-rate"])}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    strategy.log_participation_summary(num_rounds=num_rounds)
    results_logger.close()

    if bool(context.run_config["save-model"]):
        # Whatever the task chose to federate. For a from-scratch task that is
        # the full model; for a LoRA task it is only the adapters, which are
        # useless without reloading the pretrained base they were trained on.
        final_state_dict = result.arrays.to_torch_state_dict()
        torch.save(final_state_dict, "final_model.pt")
