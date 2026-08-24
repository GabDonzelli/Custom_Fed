"""Generic Flower ClientApp that delegates task-specific work."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.strategy.grouped_fedavg import PARTITION_ID_KEY
from pytorchexample.tasks.registry import get_task

app = ClientApp()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _read_partition_config(context: Context) -> tuple[int, int]:
    """Read and validate the partition identity assigned by Flower."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    configured_num_partitions = int(context.run_config["num-partitions"])

    if num_partitions != configured_num_partitions:
        raise ValueError(
            "The number of SuperNodes does not match 'num-partitions': "
            f"Flower provided {num_partitions}, configuration expects "
            f"{configured_num_partitions}."
        )
    return partition_id, num_partitions


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the selected task on one dataset partition."""
    task = get_task(str(context.run_config["task-name"]))
    model = task.create_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    partition_id, num_partitions = _read_partition_config(context)
    batch_size = int(context.run_config["batch-size"])
    trainloader, _ = task.load_partition_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
    )

    train_metrics = task.train(
        model=model,
        trainloader=trainloader,
        epochs=int(context.run_config["local-epochs"]),
        learning_rate=float(msg.content["config"]["lr"]),
        device=DEVICE,
    )

    model_record = ArrayRecord(model.state_dict())
    train_metrics["num-examples"] = len(trainloader.dataset)
    train_metrics[PARTITION_ID_KEY] = partition_id
    metric_record = MetricRecord(train_metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global model on one local validation partition."""
    task = get_task(str(context.run_config["task-name"]))
    model = task.create_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    partition_id, num_partitions = _read_partition_config(context)
    batch_size = int(context.run_config["batch-size"])
    _, validationloader = task.load_partition_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
    )
    evaluation_metrics = task.evaluate(model, validationloader, DEVICE)
    local_metrics = {
        f"eval_{metric_name}": metric_value
        for metric_name, metric_value in evaluation_metrics.items()
    }
    local_metrics["num-examples"] = len(validationloader.dataset)
    metric_record = MetricRecord(local_metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)