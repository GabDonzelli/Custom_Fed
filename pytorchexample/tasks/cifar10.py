"""CIFAR-10 model, data loading, training, and evaluation."""

import torch
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Cifar10Net(nn.Module):
    """Small convolutional network used by the Flower quickstart example."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images."""
        outputs = self.pool(F.relu(self.conv1(inputs)))
        outputs = self.pool(F.relu(self.conv2(outputs)))
        outputs = torch.flatten(outputs, 1)
        outputs = F.relu(self.fc1(outputs))
        outputs = F.relu(self.fc2(outputs))
        return self.fc3(outputs)


class Cifar10Task:
    """Implement the generic federated task interface for CIFAR-10."""

    def __init__(self) -> None:
        self._datasets: dict[int, FederatedDataset] = {}
        self._transform = Compose(
            [
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def create_model(self) -> nn.Module:
        """Create a fresh CIFAR-10 model."""
        return Cifar10Net()

    def _get_dataset(self, num_partitions: int) -> FederatedDataset:
        """Create one cached FederatedDataset per partition count."""
        if num_partitions not in self._datasets:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            self._datasets[num_partitions] = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        return self._datasets[num_partitions]

    def _apply_transforms(self, batch: dict) -> dict:
        """Convert CIFAR-10 images to normalized PyTorch tensors."""
        batch["img"] = [self._transform(image) for image in batch["img"]]
        return batch

    def load_partition_data(
        self,
        partition_id: int,
        num_partitions: int,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Load and split one IID CIFAR-10 partition."""
        dataset = self._get_dataset(num_partitions)
        partition = dataset.load_partition(partition_id)
        split = partition.train_test_split(test_size=0.2, seed=42)
        split = split.with_transform(self._apply_transforms)

        trainloader = DataLoader(split["train"], batch_size=batch_size, shuffle=True)
        validationloader = DataLoader(
            split["test"], batch_size=batch_size, shuffle=False
        )
        return trainloader, validationloader

    def load_centralized_data(
        self,
        num_partitions: int,
        batch_size: int,
    ) -> DataLoader:
        """Load the centralized CIFAR-10 test split."""
        dataset = self._get_dataset(num_partitions)
        testset = dataset.load_split("test")
        testset = testset.with_transform(self._apply_transforms)
        return DataLoader(testset, batch_size=batch_size, shuffle=False)

    def train(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        epochs: int,
        learning_rate: float,
        device: torch.device,
    ) -> dict[str, float]:
        """Train the model with SGD and return CIFAR-10 training metrics."""
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        total_loss = 0.0
        total_examples = 0
        for _ in range(epochs):
            for batch in trainloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                current_batch_size = labels.size(0)
                total_loss += loss.item() * current_batch_size
                total_examples += current_batch_size

        return {"train_loss": total_loss / total_examples}

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        """Evaluate the model and return CIFAR-10 evaluation metrics."""
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                current_batch_size = labels.size(0)
                total_loss += loss.item() * current_batch_size
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_examples += current_batch_size

        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }