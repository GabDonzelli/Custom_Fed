"""CIFAR-10 by federated linear probing of a frozen ImageNet ResNet-18.

The vision counterpart of :mod:`pytorchexample.tasks.stackexchange_pretrained`:
same idea -- freeze a pretrained backbone, federate only a small trainable head
-- applied to images instead of text.

Why a linear probe and not full fine-tuning
-------------------------------------------
A pretrained *vision* classifier cannot be evaluated zero-shot on CIFAR-10: its
head predicts 1000 ImageNet classes, not CIFAR-10's 10, so the head has to be
replaced and a fresh head is random. The pretrained knowledge lives in the
features, and the honest measurement of "what does ImageNet already know about
CIFAR-10" is a linear probe -- see ``pretrained_cifar.py``, which measures
**83.8%** centrally with exactly this setup (2000 train / 1000 test images).

That also happens to be the only affordable option here: this machine has no
GPU, and the frozen backbone runs at ~12 images/s end to end (JPEG decode plus
resize to 224 plus forward). Fine-tuning the 11M backbone weights on 10 clients
for 20 rounds is out of reach on CPU; training a 5,130-parameter head is not.

What is federated
-----------------
Only ``nn.Linear(512, 10)`` -- 5,130 floats per client per round against the
11.2M of the full ResNet-18. Same communication argument as LoRA.

Why features are L2-normalized per sample
-----------------------------------------
The reference probe standardizes features with the *dataset's* mean and
variance. That cannot survive FedAvg: each client would standardize with its
own statistics, so the heads being averaged would no longer speak about the
same input space. Per-sample L2 normalization depends on nothing but the image
itself, so every client's head shares one input space -- and it trains better
here anyway. Measured on 2000/1000 images, 30 epochs, one machine:

    raw features    + SGD(lr=0.1, m=0.9)   79.9%
    L2 per sample   + SGD(lr=0.1, m=0.9)   82.2%
    L2 per sample   + SGD(lr=1.0, m=0.9)   84.8%   <- what this task expects
    L2 per sample   + Adam(lr=1e-3)        81.6%

Note the learning rate: L2-normalized features have norm 1, so the logits are
small and the head wants a *large* step. Use ``learning-rate = 1.0`` for this
task, not the 0.1 of the from-scratch CIFAR-10 task.

Caveat: this is not an equal-data comparison against ``cifar10``
----------------------------------------------------------------
``Cifar10Task`` trains its small CNN on the whole partition (4,000 images per
client at 10 partitions). Here each client is capped to
``MAX_IMAGES_PER_CLIENT`` = 500, because every image has to pass through the
frozen backbone. The cap handicaps *this* task, so the gap it still shows over
the from-scratch CNN is a lower bound on what pretraining buys.
"""

import time
from logging import INFO

import torch
from flwr.common import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# ImageNet normalization: what the pretrained weights were trained to expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The resolution the ImageNet weights expect. Dropping to 64 would roughly
# double throughput but costs a large chunk of the probe accuracy, and 224 is
# what the 83.8% reference number in `pretrained_cifar.py` was measured at.
INPUT_RESOLUTION = 224

# ResNet-18's penultimate feature width, and the head that rides on top of it.
FEATURE_DIM = 512
NUM_CLASSES = 10

# Caps on how many images ever cross the frozen backbone. Purely a CPU budget:
# at ~12 img/s these add up to ~9 minutes of one-time feature extraction for
# the whole federation, against ~70 minutes for uncapped partitions.
MAX_IMAGES_PER_CLIENT = 500
MAX_VALIDATION_IMAGES_PER_CLIENT = 125
MAX_CENTRALIZED_IMAGES = 1000

# Seed for the deterministic subsampling above, so a client sees the same
# images in every round and across runs.
SUBSAMPLE_SEED = 42


class LinearHead(nn.Module):
    """The only trainable part: a linear classifier over frozen features."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify a batch of already-extracted, already-normalized features."""
        return self.classifier(features)


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    """Scale each feature vector to unit norm, independently of every other."""
    return features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)


class Cifar10PretrainedTask:
    """Federated linear probe on a frozen ImageNet-pretrained ResNet-18."""

    def __init__(self) -> None:
        self._datasets: dict[int, FederatedDataset] = {}
        self._backbone: nn.Module | None = None
        # Features are extracted once per process and reused for every round:
        # the backbone is frozen, so a given image's features never change.
        # This is what makes 20 rounds cost about as much as one.
        self._partition_cache: dict[
            tuple[int, int], tuple[TensorDataset, TensorDataset]
        ] = {}
        self._centralized_cache: dict[int, TensorDataset] = {}
        self._transform = Compose(
            [
                Resize(INPUT_RESOLUTION),
                ToTensor(),
                Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    # ---------------------------------------------------------------- model

    def create_model(self) -> nn.Module:
        """Create a fresh, randomly initialized linear head.

        Round 0 of a run is therefore chance accuracy (10%), not the probe's
        83.8%: unlike LoRA -- whose zero-initialized ``B`` makes round 0 exactly
        the pretrained model -- a vision head has no zero-shot state to inherit.
        """
        return LinearHead()

    def get_federated_arrays(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Communicate the head only; the backbone is frozen and never sent."""
        return model.state_dict()

    def load_federated_arrays(
        self, model: nn.Module, arrays: dict[str, torch.Tensor]
    ) -> None:
        """Load a head state dict, requiring it to cover the head exactly."""
        model.load_state_dict(arrays)

    # ------------------------------------------------------------- features

    def _get_backbone(self) -> nn.Module:
        """Load the pretrained ResNet-18 once and strip its ImageNet head."""
        if self._backbone is None:
            import torchvision

            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            model = torchvision.models.resnet18(weights=weights)
            # Everything but the 1000-class fc: the output is the 512-d
            # average-pooled feature map this task classifies.
            backbone = nn.Sequential(*list(model.children())[:-1])
            backbone.eval()
            for parameter in backbone.parameters():
                parameter.requires_grad = False
            self._backbone = backbone
        return self._backbone

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
        """Resize to 224 and apply ImageNet normalization."""
        batch["img"] = [self._transform(image) for image in batch["img"]]
        return batch

    @torch.no_grad()
    def _extract_features(self, split, label: str) -> TensorDataset:
        """Push a split once through the frozen backbone, L2-normalizing it."""
        backbone = self._get_backbone()
        loader = DataLoader(
            split.with_transform(self._apply_transforms), batch_size=32
        )

        started = time.time()
        feature_batches, label_batches = [], []
        for batch in loader:
            feature_batches.append(backbone(batch["img"]).flatten(1))
            label_batches.append(batch["label"])

        features = l2_normalize(torch.cat(feature_batches))
        labels = torch.cat(label_batches)
        log(
            INFO,
            "%s: extracted %d features in %.0fs (frozen backbone, cached from "
            "here on)",
            label,
            features.size(0),
            time.time() - started,
        )
        return TensorDataset(features, labels)

    @staticmethod
    def _subsample(split, limit: int | None, seed: int):
        """Deterministically shrink a split to `limit` rows, or leave it be."""
        if limit is None or len(split) <= limit:
            return split
        return split.shuffle(seed=seed).select(range(limit))

    # ---------------------------------------------------------------- data

    def load_partition_data(
        self,
        partition_id: int,
        num_partitions: int,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Load one IID partition as cached, frozen-backbone features."""
        cache_key = (partition_id, num_partitions)
        if cache_key not in self._partition_cache:
            partition = self._get_dataset(num_partitions).load_partition(partition_id)
            split = partition.train_test_split(test_size=0.2, seed=42)
            seed = SUBSAMPLE_SEED + partition_id
            self._partition_cache[cache_key] = (
                self._extract_features(
                    self._subsample(split["train"], MAX_IMAGES_PER_CLIENT, seed),
                    f"partition {partition_id} train",
                ),
                self._extract_features(
                    self._subsample(
                        split["test"], MAX_VALIDATION_IMAGES_PER_CLIENT, seed
                    ),
                    f"partition {partition_id} validation",
                ),
            )

        train_features, validation_features = self._partition_cache[cache_key]
        return (
            DataLoader(train_features, batch_size=batch_size, shuffle=True),
            DataLoader(validation_features, batch_size=batch_size, shuffle=False),
        )

    def load_centralized_data(
        self,
        num_partitions: int,
        batch_size: int,
    ) -> DataLoader:
        """Load the CIFAR-10 test split as cached, frozen-backbone features."""
        if num_partitions not in self._centralized_cache:
            testset = self._get_dataset(num_partitions).load_split("test")
            self._centralized_cache[num_partitions] = self._extract_features(
                self._subsample(testset, MAX_CENTRALIZED_IMAGES, SUBSAMPLE_SEED),
                "centralized test",
            )
        return DataLoader(
            self._centralized_cache[num_partitions],
            batch_size=batch_size,
            shuffle=False,
        )

    # ------------------------------------------------------- train/evaluate

    def train(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        epochs: int,
        learning_rate: float,
        device: torch.device,
    ) -> dict[str, float]:
        """Train the head with SGD. Expects `learning_rate` near 1.0."""
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        total_loss = 0.0
        total_examples = 0
        for _ in range(epochs):
            for features, labels in trainloader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                loss = criterion(model(features), labels)
                loss.backward()
                optimizer.step()

                current_batch_size = labels.size(0)
                total_loss += loss.item() * current_batch_size
                total_examples += current_batch_size

        return {"train_loss": total_loss / max(total_examples, 1)}

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        """Return the head's loss and accuracy on frozen features."""
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                current_batch_size = labels.size(0)
                total_loss += loss.item() * current_batch_size
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_examples += current_batch_size

        return {
            "loss": total_loss / max(total_examples, 1),
            "accuracy": total_correct / max(total_examples, 1),
        }
