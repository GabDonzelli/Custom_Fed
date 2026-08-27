"""Stack Exchange next-word-prediction task, partitioned by author (non-IID).

Each simulated client is one real Stack Exchange author, so partitions are
naturally non-IID (different people write about different topics/sites).

The Hub copy of ``HuggingFaceH4/stack-exchange-preferences`` does not support
cheap row-slicing: even a `datasets.load_dataset(..., split="train[:2000]")`
call downloads full multi-gigabyte parquet shards. This module instead reads
the dataset through the streaming API and stops once ``MAX_QUESTIONS_SCANNED``
questions have been scanned, which keeps the download small and bounded.
"""

import html
import re
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATASET_NAME = "HuggingFaceH4/stack-exchange-preferences"

# How many questions to scan from the stream. Larger values find more
# distinct authors (needed for a large num-partitions) at the cost of a
# longer, but still bounded and network-cheap, initial scan.
MAX_QUESTIONS_SCANNED = 20_000

# Authors reserved for centralized evaluation only (never used for training),
# so ServerApp-side evaluation measures generalization to unseen authors.
HELD_OUT_AUTHORS = 20

# An author needs at least this many qualifying answers to be usable as a
# client (so its local 80/20 train/validation split is never empty).
MIN_ANSWERS_PER_AUTHOR = 2

# Answers shorter than this (in tokens) are dropped as too weak a signal for
# next-word prediction.
MIN_TOKENS_PER_EXAMPLE = 4

SEQ_LEN = 32  # tokens per training example
VOCAB_SIZE = 5_000  # includes <pad> and <unk>
PAD_ID = 0
UNK_ID = 1

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _clean_text(raw_html: str) -> str:
    """Strip HTML markup/entities from a raw Stack Exchange answer body."""
    return _HTML_TAG_RE.sub(" ", html.unescape(raw_html))


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only whitespace tokenization."""
    return _TOKEN_RE.findall(text.lower())


def _stream_author_answers(max_questions: int) -> list[tuple[int, list[str]]]:
    """Stream questions from the Hub and flatten to (author_id, tokens) pairs."""
    stream = load_dataset(DATASET_NAME, split="train", streaming=True)
    examples: list[tuple[int, list[str]]] = []
    for question_index, question in enumerate(stream):
        if question_index >= max_questions:
            break
        for answer in question["answers"]:
            tokens = _tokenize(_clean_text(answer["text"]))
            if len(tokens) >= MIN_TOKENS_PER_EXAMPLE:
                examples.append((int(answer["author_id"]), tokens))
    return examples


def _select_partitions(
    examples: list[tuple[int, list[str]]],
    num_partitions: int,
    held_out_authors: int = HELD_OUT_AUTHORS,
    min_answers_per_author: int = MIN_ANSWERS_PER_AUTHOR,
) -> tuple[dict[int, int], list[int]]:
    """Rank qualifying authors by activity and split into training/held-out.

    Returns a mapping from author_id to a 0-based partition_id for the
    `num_partitions` most active qualifying authors (the simulated clients),
    plus the author_ids reserved for centralized evaluation only.

    Pure function (no I/O), so it is unit-testable without network access.
    """
    counts = Counter(author_id for author_id, _ in examples)
    eligible_authors = [
        author_id
        for author_id, count in counts.items()
        if count >= min_answers_per_author
    ]
    ranked_authors = sorted(
        eligible_authors, key=lambda author_id: (-counts[author_id], author_id)
    )

    required = num_partitions + held_out_authors
    if len(ranked_authors) < required:
        raise ValueError(
            f"Only {len(ranked_authors)} authors with >= {min_answers_per_author} "
            f"qualifying answers were found, but {required} are needed "
            f"({num_partitions} partitions + {held_out_authors} held out for "
            "centralized evaluation). Increase MAX_QUESTIONS_SCANNED in "
            "pytorchexample/tasks/stackexchange.py."
        )

    training_authors = ranked_authors[:num_partitions]
    held_out = ranked_authors[num_partitions:required]
    author_to_partition = {
        author_id: partition_id
        for partition_id, author_id in enumerate(training_authors)
    }
    return author_to_partition, held_out


class _TokenSequenceDataset(Dataset):
    """Fixed-length token id sequences for next-word-prediction training."""

    def __init__(self, sequences: list[list[int]]) -> None:
        self._sequences = sequences

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self._sequences[index], dtype=torch.long)


class StackExchangeLSTM(nn.Module):
    """Small word-level LSTM language model."""

    def __init__(
        self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for every position in the sequence."""
        embedded = self.embedding(token_ids)
        hidden_states, _ = self.lstm(embedded)
        return self.output(hidden_states)


class StackExchangeTask:
    """Non-IID next-word-prediction task: one Stack Exchange author per client."""

    def __init__(self) -> None:
        self._author_examples: dict[int, list[list[str]]] = {}
        self._held_out_tokens: list[list[str]] = []
        self._vocab: dict[str, int] | None = None

    def _ensure_dataset(self, num_partitions: int) -> None:
        """Scan and cache the flattened, author-partitioned dataset once."""
        if self._vocab is not None and len(self._author_examples) == num_partitions:
            return

        examples = _stream_author_answers(MAX_QUESTIONS_SCANNED)
        author_to_partition, held_out_authors = _select_partitions(
            examples, num_partitions
        )
        held_out_set = set(held_out_authors)

        author_examples: dict[int, list[list[str]]] = {
            partition_id: [] for partition_id in author_to_partition.values()
        }
        held_out_tokens: list[list[str]] = []
        for author_id, tokens in examples:
            if author_id in author_to_partition:
                author_examples[author_to_partition[author_id]].append(tokens)
            elif author_id in held_out_set:
                held_out_tokens.append(tokens)

        vocab_counts = Counter(
            token
            for tokens_list in author_examples.values()
            for tokens in tokens_list
            for token in tokens
        )
        vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
        for token, _ in vocab_counts.most_common(VOCAB_SIZE - len(vocab)):
            vocab[token] = len(vocab)

        self._author_examples = author_examples
        self._held_out_tokens = held_out_tokens
        self._vocab = vocab

    def _encode(self, tokens: list[str]) -> list[int]:
        """Numericalize and pad/truncate one token sequence to SEQ_LEN."""
        assert self._vocab is not None
        ids = [self._vocab.get(token, UNK_ID) for token in tokens[:SEQ_LEN]]
        ids += [PAD_ID] * (SEQ_LEN - len(ids))
        return ids

    def create_model(self) -> nn.Module:
        """Create a fresh Stack Exchange language model."""
        return StackExchangeLSTM()

    def load_partition_data(
        self,
        partition_id: int,
        num_partitions: int,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Load one author's answers, split 80/20 into train/validation."""
        self._ensure_dataset(num_partitions)
        tokens_list = self._author_examples[partition_id]

        split_point = max(1, int(len(tokens_list) * 0.8))
        train_sequences = [self._encode(t) for t in tokens_list[:split_point]]
        val_sequences = [self._encode(t) for t in tokens_list[split_point:]]

        trainloader = DataLoader(
            _TokenSequenceDataset(train_sequences), batch_size=batch_size, shuffle=True
        )
        validationloader = DataLoader(
            _TokenSequenceDataset(val_sequences), batch_size=batch_size, shuffle=False
        )
        return trainloader, validationloader

    def load_centralized_data(
        self,
        num_partitions: int,
        batch_size: int,
    ) -> DataLoader:
        """Load the held-out authors' answers (never seen during training)."""
        self._ensure_dataset(num_partitions)
        sequences = [self._encode(tokens) for tokens in self._held_out_tokens]
        return DataLoader(
            _TokenSequenceDataset(sequences), batch_size=batch_size, shuffle=False
        )

    def train(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        epochs: int,
        learning_rate: float,
        device: torch.device,
    ) -> dict[str, float]:
        """Train the language model with SGD and return the mean token loss."""
        model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        total_loss = 0.0
        total_tokens = 0
        for _ in range(epochs):
            for batch in trainloader:
                batch = batch.to(device)
                inputs, targets = batch[:, :-1], batch[:, 1:]

                optimizer.zero_grad()
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=PAD_ID,
                )
                loss.backward()
                optimizer.step()

                num_target_tokens = max(int((targets != PAD_ID).sum().item()), 1)
                total_loss += loss.item() * num_target_tokens
                total_tokens += num_target_tokens

        return {"train_loss": total_loss / total_tokens}

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        """Evaluate next-word-prediction loss and token-level accuracy."""
        model.to(device)
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=PAD_ID,
                )

                mask = targets != PAD_ID
                num_target_tokens = max(int(mask.sum().item()), 1)
                predictions = logits.argmax(dim=-1)
                total_correct += int(((predictions == targets) & mask).sum().item())
                total_loss += loss.item() * num_target_tokens
                total_tokens += num_target_tokens

        return {
            "loss": total_loss / total_tokens,
            "accuracy": total_correct / total_tokens,
        }
