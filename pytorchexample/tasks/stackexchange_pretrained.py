"""Stack Exchange next-word prediction by federated LoRA fine-tuning of DistilGPT-2.

Same non-IID partitioning as :mod:`pytorchexample.tasks.stackexchange` -- one
real Stack Exchange author per client, reusing that module's scan, cache and
author-ranking helpers -- but the model is a *pretrained* language model instead
of an LSTM trained from scratch.

Why this fixes the problem the from-scratch task has
----------------------------------------------------
``StackExchangeLSTM`` has 2.32M parameters against 306k training tokens (7.6x),
which is why it peaks at round ~5 and then degrades for the rest of the run --
see the vault note "Diagnóstico - Accuracy cai e loss sobe a cada rodada".
Making the model *bigger* would make that worse.

A pretrained model escapes the trade-off from a different direction: it is not
estimating its parameters from those 306k tokens at all. It already models
English; the local data only has to adapt it. Only the LoRA adapters are
trained, so the number of parameters actually fitted to the local corpus is
small (~150k) even though the model is 82M.

Measured starting point (2026-09-02): DistilGPT-2 scores **16.89%** whole-word
accuracy zero-shot on these same held-out authors, already above the 15.0% peak
the from-scratch LSTM reaches after 5 federated rounds. Because LoRA's ``B`` is
zero-initialized, **round 0 of a run here is exactly that zero-shot model**, so
the run's own round-0 metric is the baseline to compare later rounds against.

Caveat carried over from the corpus
-----------------------------------
The cached scan stores lowercased, punctuation-free word tokens (see
``_TOKEN_RE`` in the from-scratch module), and this task rebuilds text by
joining them with spaces. That *handicaps* a pretrained model, which was trained
on natural text. Re-scanning the dataset to keep raw text would raise these
numbers, but it needs network access and a new cache, so it is left as a
follow-up.
"""

import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pytorchexample.tasks.lora import (
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    trainable_parameters,
)
from pytorchexample.tasks.stackexchange import (
    MAX_QUESTIONS_SCANNED,
    _select_partitions,
    _stream_author_answers,
)

MODEL_NAME = "distilgpt2"

# LoRA is applied to the fused query/key/value projection of every attention
# block, the standard target for GPT-2 style models.
LORA_TARGET_SUFFIXES = ("attn.c_attn",)
LORA_RANK = 8
LORA_ALPHA = 16.0

# BPE tokens per training example. Answers are split into consecutive blocks of
# this length rather than truncated, so no tokens are thrown away -- the
# from-scratch task discards 71% of the corpus by truncating at 32 words.
BLOCK_SIZE = 64

# Blocks shorter than this after chunking are dropped rather than padded: a
# handful of real tokens is too weak a signal to be worth the padding.
MIN_BLOCK_TOKENS = 16

# Ignored by cross_entropy, marking padding positions in the labels.
IGNORE_INDEX = -100

# Caps on how much data each client and the centralized evaluator actually use.
# Set to None to use everything.
#
# These exist for two reasons at once:
#
# 1. Cost. DistilGPT-2 trains at ~7 blocks/s on CPU, and the 10 partitions hold
#    13,876 training blocks; an uncapped round at 5 local epochs takes ~2.75h.
#
# 2. Balance. Authors are ranked by activity, so partition 0 is the single most
#    prolific author and holds 7,885 blocks -- 56.8% of all training data. Since
#    FedAvg weights by example count, the global model is otherwise essentially
#    that one author's model. Capping is the fix the vault note
#    "Achados - Pendências e Bugs" lists as "limitar exemplos por cliente".
#
# Note the side effect: with every client capped to the same ceiling, the clients
# become near-uniform in size, so weighted FedAvg approaches unweighted FedAvg.
MAX_BLOCKS_PER_CLIENT = 250
MAX_VALIDATION_BLOCKS_PER_CLIENT = 100
MAX_CENTRALIZED_BLOCKS = 800

# Seed for the deterministic subsampling the caps above perform, so a client
# trains on the same subset in every round and across runs.
SUBSAMPLE_SEED = 42


class _BlockDataset(Dataset):
    """Fixed-length BPE blocks with padded positions masked out of the labels."""

    def __init__(self, blocks: list[tuple[list[int], list[int]]]) -> None:
        self._blocks = blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, labels = self._blocks[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def chunk_into_blocks(
    token_ids: list[int],
    block_size: int = BLOCK_SIZE,
    min_block_tokens: int = MIN_BLOCK_TOKENS,
    pad_id: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Split one answer's token ids into padded, label-masked blocks.

    Pure function so the chunking can be unit-tested without a tokenizer.
    """
    blocks = []
    for start in range(0, len(token_ids), block_size):
        chunk = token_ids[start : start + block_size]
        if len(chunk) < min_block_tokens:
            continue
        padding = block_size - len(chunk)
        input_ids = chunk + [pad_id] * padding
        labels = chunk + [IGNORE_INDEX] * padding
        blocks.append((input_ids, labels))
    return blocks


def cap_blocks(
    blocks: list[tuple[list[int], list[int]]],
    limit: int | None,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """Deterministically subsample `blocks` down to `limit`, or return as-is.

    Seeded on the caller's key so the same client keeps the same subset in
    every round -- a resample each round would make local data drift between
    rounds, which is not what the experiment intends to vary.
    """
    if limit is None or len(blocks) <= limit:
        return blocks
    return random.Random(seed).sample(blocks, limit)


class StackExchangePretrainedTask:
    """Federated LoRA fine-tuning of DistilGPT-2, one author per client."""

    def __init__(self) -> None:
        self._author_examples: dict[int, list[list[str]]] = {}
        self._held_out_tokens: list[list[str]] = []
        self._tokenizer = None

    def _get_tokenizer(self):
        """Load the GPT-2 BPE tokenizer once, with a usable pad token."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            # GPT-2 ships no pad token. Reusing eos is the standard workaround;
            # padded positions are masked out of the labels regardless, so the
            # choice never reaches the loss.
            tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer = tokenizer
        return self._tokenizer

    def _ensure_dataset(self, num_partitions: int) -> None:
        """Scan and cache the author-partitioned dataset once."""
        if self._author_examples and len(self._author_examples) == num_partitions:
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

        self._author_examples = author_examples
        self._held_out_tokens = held_out_tokens

    def _build_blocks(
        self, token_lists: list[list[str]]
    ) -> list[tuple[list[int], list[int]]]:
        """Tokenize word sequences with BPE and chunk them into blocks."""
        tokenizer = self._get_tokenizer()
        pad_id = tokenizer.pad_token_id
        blocks = []
        for tokens in token_lists:
            # verbose=False silences the tokenizer's "longer than model_max_length"
            # warning: a whole answer can exceed GPT-2's 1024-token context, but
            # chunk_into_blocks immediately splits it into BLOCK_SIZE pieces, so
            # nothing that long is ever fed to the model.
            encoded = tokenizer(
                " ".join(tokens), add_special_tokens=False, verbose=False
            ).input_ids
            blocks.extend(chunk_into_blocks(encoded, pad_id=pad_id))
        return blocks

    def create_model(self) -> nn.Module:
        """Load pretrained DistilGPT-2 and attach freshly zeroed LoRA adapters.

        The pretrained weights come from the local HuggingFace cache on every
        call and are frozen; only the adapters are trainable, and they travel
        over the wire. Because LoRA's ``B`` starts at zero, the model returned
        here is numerically identical to plain pretrained DistilGPT-2.
        """
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        inject_lora(
            model,
            target_suffixes=LORA_TARGET_SUFFIXES,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
        )
        return model

    def get_federated_arrays(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Communicate only the LoRA adapters, not the frozen 82M base."""
        return lora_state_dict(model)

    def load_federated_arrays(
        self, model: nn.Module, arrays: dict[str, torch.Tensor]
    ) -> None:
        """Load adapters into a model whose base weights are already pretrained."""
        load_lora_state_dict(model, arrays)

    def load_partition_data(
        self,
        partition_id: int,
        num_partitions: int,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Load one author's answers, split 80/20 into train/validation."""
        self._ensure_dataset(num_partitions)
        token_lists = self._author_examples[partition_id]

        split_point = max(1, int(len(token_lists) * 0.8))
        train_blocks = cap_blocks(
            self._build_blocks(token_lists[:split_point]),
            MAX_BLOCKS_PER_CLIENT,
            SUBSAMPLE_SEED + partition_id,
        )
        validation_blocks = cap_blocks(
            self._build_blocks(token_lists[split_point:]),
            MAX_VALIDATION_BLOCKS_PER_CLIENT,
            SUBSAMPLE_SEED + partition_id,
        )

        trainloader = DataLoader(
            _BlockDataset(train_blocks), batch_size=batch_size, shuffle=True
        )
        validationloader = DataLoader(
            _BlockDataset(validation_blocks), batch_size=batch_size, shuffle=False
        )
        return trainloader, validationloader

    def load_centralized_data(
        self,
        num_partitions: int,
        batch_size: int,
    ) -> DataLoader:
        """Load the held-out authors' answers (never seen during training)."""
        self._ensure_dataset(num_partitions)
        blocks = cap_blocks(
            self._build_blocks(self._held_out_tokens),
            MAX_CENTRALIZED_BLOCKS,
            SUBSAMPLE_SEED,
        )
        return DataLoader(_BlockDataset(blocks), batch_size=batch_size, shuffle=False)

    @staticmethod
    def _shifted_loss(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (loss, predictions, targets) for next-token prediction."""
        shifted_logits = logits[:, :-1]
        shifted_labels = labels[:, 1:]
        loss = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        return loss, shifted_logits.argmax(dim=-1), shifted_labels

    def train(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        epochs: int,
        learning_rate: float,
        device: torch.device,
    ) -> dict[str, float]:
        """Fine-tune only the LoRA adapters and return the mean token loss.

        Uses AdamW rather than the SGD of the from-scratch tasks: with ``B``
        zero-initialized, LoRA adapters train poorly under plain SGD, and AdamW
        is what the LoRA literature uses. Expect ``learning-rate`` around 5e-4
        here, not the 0.1 the from-scratch tasks want.
        """
        model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            trainable_parameters(model), lr=learning_rate
        )

        total_loss = 0.0
        total_tokens = 0
        for _ in range(epochs):
            for batch in trainloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                logits = model(input_ids=input_ids).logits
                loss, _, targets = self._shifted_loss(logits, labels)
                loss.backward()
                # Cheap, and removes the NaN risk left open in the vault notes.
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
                optimizer.step()

                num_target_tokens = max(
                    int((targets != IGNORE_INDEX).sum().item()), 1
                )
                total_loss += loss.item() * num_target_tokens
                total_tokens += num_target_tokens

        return {"train_loss": total_loss / max(total_tokens, 1)}

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        """Return next-token loss, accuracy and perplexity."""
        model.to(device)
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids=input_ids).logits
                loss, predictions, targets = self._shifted_loss(logits, labels)

                mask = targets != IGNORE_INDEX
                num_target_tokens = max(int(mask.sum().item()), 1)
                total_correct += int(((predictions == targets) & mask).sum().item())
                total_loss += loss.item() * num_target_tokens
                total_tokens += num_target_tokens

        mean_loss = total_loss / max(total_tokens, 1)
        return {
            "loss": mean_loss,
            "accuracy": total_correct / max(total_tokens, 1),
            "perplexity": float(torch.exp(torch.tensor(mean_loss))),
        }
