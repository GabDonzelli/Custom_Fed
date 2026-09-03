"""What accuracy does an ImageNet-pretrained backbone already give on CIFAR-10?

Unlike a language model, a pretrained *vision* classifier cannot be evaluated
zero-shot on CIFAR-10: its head predicts 1000 ImageNet classes, not CIFAR-10's
10. The head must be replaced, and a fresh head is random -- so true zero-shot
accuracy is chance (10%). The pretrained knowledge lives in the *features*.

So the meaningful measurement is a LINEAR PROBE: freeze the backbone, extract
features once, and train only a linear classifier on top. That number is the
honest answer to "what does the pretrained model already know about CIFAR-10",
and it is also the floor that federated fine-tuning starts from.

For reference the same probe is run on an untrained (random-weight) backbone,
which isolates how much of the accuracy comes from the ImageNet pretraining
rather than from the architecture plus the linear head.
"""

import argparse
import time

import torch
import torchvision
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# ImageNet normalization -- must match what the pretrained weights expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_loaders(resolution, num_train, num_test, batch_size):
    transform = Compose(
        [
            Resize(resolution),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    def apply_transforms(batch):
        batch["img"] = [transform(image) for image in batch["img"]]
        return batch

    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": IidPartitioner(num_partitions=1)},
    )
    train = fds.load_partition(0).shuffle(seed=42).select(range(num_train))
    test = fds.load_split("test").shuffle(seed=42).select(range(num_test))
    train = train.with_transform(apply_transforms)
    test = test.with_transform(apply_transforms)
    return (
        DataLoader(train, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


@torch.no_grad()
def extract_features(backbone, loader, label):
    """Run the frozen backbone once and cache its penultimate features."""
    backbone.eval()
    features, labels = [], []
    started = time.time()
    for index, batch in enumerate(loader):
        features.append(backbone(batch["img"]).flatten(1))
        labels.append(batch["label"])
        if index % 10 == 0:
            done = sum(f.size(0) for f in features)
            print(f"  {label}: {done} images ({time.time() - started:.0f}s)", flush=True)
    return torch.cat(features), torch.cat(labels)


def train_linear_probe(train_features, train_labels, test_features, test_labels, epochs=200):
    """Train only a linear classifier on frozen, precomputed features."""
    # Standardize features -- makes the probe converge fast and reliably.
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True) + 1e-6
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std

    torch.manual_seed(42)
    probe = nn.Linear(train_features.size(1), 10)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        criterion(probe(train_features), train_labels).backward()
        optimizer.step()

    with torch.no_grad():
        train_accuracy = (probe(train_features).argmax(1) == train_labels).float().mean().item()
        test_accuracy = (probe(test_features).argmax(1) == test_labels).float().mean().item()
    return train_accuracy, test_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    print(f"config: {vars(args)}", flush=True)

    trainloader, testloader = build_loaders(
        args.resolution, args.num_train, args.num_test, args.batch_size
    )

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    print(f"\nImageNet top-1 of these weights, as published: {weights.meta['_metrics']['ImageNet-1K']['acc@1']}%", flush=True)

    for label, model_weights in (("ImageNet-pretrained", weights), ("random init", None)):
        print(f"\n=== ResNet-18, {label} ===", flush=True)
        model = torchvision.models.resnet18(weights=model_weights)
        num_params = sum(p.numel() for p in model.parameters())
        # Drop the 1000-class ImageNet head: it cannot predict CIFAR-10 classes.
        backbone = nn.Sequential(*list(model.children())[:-1])
        print(f"{num_params:,} params (backbone feature dim 512)", flush=True)

        train_features, train_labels = extract_features(backbone, trainloader, "train")
        test_features, test_labels = extract_features(backbone, testloader, "test")

        train_accuracy, test_accuracy = train_linear_probe(
            train_features, train_labels, test_features, test_labels
        )
        print(
            f"LINEAR PROBE  train acc {train_accuracy:.4f}  "
            f"TEST ACC {test_accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
