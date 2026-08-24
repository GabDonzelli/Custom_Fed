"""Common interface implemented by every federated learning task."""

from typing import Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader


class FederatedTask(Protocol):
    """Describe the operations required by the generic Flower application."""

    def create_model(self) -> nn.Module:
        """Create a new model with the architecture required by the task."""

    def load_partition_data(
        self,
        partition_id: int,
        num_partitions: int,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Load the local train and validation data for one partition."""

    def load_centralized_data(
        self,
        num_partitions: int,
        batch_size: int,
    ) -> DataLoader:
        """Load the centralized test data used by the ServerApp."""

    def train(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        epochs: int,
        learning_rate: float,
        device: torch.device,
    ) -> dict[str, float]:
        """Train a model locally and return task-specific training metrics."""

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        """Return task-specific evaluation metrics for the supplied data."""