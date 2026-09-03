"""Task registry used by both the ClientApp and the ServerApp."""

from functools import cache

from pytorchexample.tasks.base import FederatedTask
from pytorchexample.tasks.cifar10 import Cifar10Task
from pytorchexample.tasks.stackexchange import StackExchangeTask
from pytorchexample.tasks.stackexchange_pretrained import (
    StackExchangePretrainedTask,
)

TASK_FACTORIES = {
    "cifar10": Cifar10Task,
    "stackexchange": StackExchangeTask,
    "stackexchange-pretrained": StackExchangePretrainedTask,
}


@cache
def get_task(task_name: str) -> FederatedTask:
    """Create and cache the task selected in the run configuration."""
    normalized_name = task_name.strip().lower()
    try:
        task_factory = TASK_FACTORIES[normalized_name]
    except KeyError as exc:
        available_tasks = ", ".join(sorted(TASK_FACTORIES))
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {available_tasks}."
        ) from exc
    return task_factory()