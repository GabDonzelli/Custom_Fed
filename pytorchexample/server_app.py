"""ServerApp entry point for proportional grouped FedAvg."""
from logging import INFO

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from flwr.serverapp.strategy import FedAvg
from pytorchexample.tasks.registry import get_task

app = ServerApp()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run federated training with four proportional client groups."""
    task_name = str(context.run_config["task-name"])
    num_partitions = int(context.run_config["num-partitions"])
    num_groups = int(context.run_config["num-groups"])
    batch_size = int(context.run_config["batch-size"])

    task = get_task(task_name)
    global_model = task.create_model()
    initial_arrays = ArrayRecord(global_model.state_dict())
    centralized_testloader = task.load_centralized_data(
        num_partitions=num_partitions,
        batch_size=batch_size,
    )

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_train_nodes=num_partitions,
        min_evaluate_nodes=num_partitions,
        min_available_nodes=num_partitions,
        weighted_by_key="num-examples",
    )

    def global_evaluate(
        server_round: int,
        arrays: ArrayRecord,
    ) -> MetricRecord:
        """Evaluate the current global model on centralized test data."""
        model = task.create_model()
        model.load_state_dict(arrays.to_torch_state_dict())
        evaluation_metrics = task.evaluate(model, centralized_testloader, DEVICE)
        return MetricRecord(evaluation_metrics)

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": float(context.run_config["learning-rate"])}),
        num_rounds=int(context.run_config["num-server-rounds"]),
        evaluate_fn=global_evaluate,
    )

    if bool(context.run_config["save-model"]):
        final_state_dict = result.arrays.to_torch_state_dict()
        torch.save(final_state_dict, "final_model.pt")
