"""Zero-shot next-word accuracy of a pretrained LM on the held-out Stack Exchange authors.

Measures two different things, because they answer different questions:

1. BPE-token top-1 accuracy -- the natural metric for GPT-2, over its own 50257
   subword vocabulary. NOT comparable to the current task's 15%.
2. Word-level top-1 accuracy -- greedily decode the next *whole word* and compare
   with the true next word. This IS comparable to the current task, which
   predicts whole words over a 5000-word vocabulary.

Both are zero-shot: no fine-tuning, no gradient step, weights straight from the Hub.
"""

import argparse
import sys

import torch

sys.path.insert(0, ".")
from pytorchexample.tasks.stackexchange import SEQ_LEN, StackExchangeTask  # noqa: E402


def load_heldout_word_sequences(num_partitions, max_sequences):
    """Reconstruct held-out author text as whitespace-joined word sequences."""
    task = StackExchangeTask()
    task._ensure_dataset(num_partitions)
    sequences = [
        tokens[:SEQ_LEN] for tokens in task._held_out_tokens if len(tokens) >= 8
    ]
    return sequences[:max_sequences], task


@torch.no_grad()
def bpe_token_accuracy(model, tokenizer, sequences):
    """Top-1 accuracy over GPT-2's own subword vocabulary."""
    total_correct = total = 0
    for words in sequences:
        ids = tokenizer(" ".join(words), return_tensors="pt").input_ids
        if ids.size(1) < 2:
            continue
        logits = model(ids).logits
        predictions = logits[0, :-1].argmax(dim=-1)
        targets = ids[0, 1:]
        total_correct += (predictions == targets).sum().item()
        total += targets.numel()
    return total_correct / max(total, 1), total


@torch.no_grad()
def word_level_accuracy(model, tokenizer, sequences, context_words=8):
    """Greedily decode the next whole word and compare with the true next word.

    For every position from `context_words` onward, feed the preceding words and
    generate until a word boundary, then compare the decoded word with the truth.
    """
    total_correct = total = 0
    for words in sequences:
        for split in range(context_words, len(words)):
            context = " ".join(words[:split])
            true_word = words[split]

            ids = tokenizer(context, return_tensors="pt").input_ids
            generated = []
            past = None
            current = ids
            # A word is at most a handful of BPE pieces; 6 is generous.
            for _ in range(6):
                output = model(current, past_key_values=past, use_cache=True)
                past = output.past_key_values
                next_id = output.logits[0, -1].argmax().item()
                piece = tokenizer.decode([next_id])
                generated.append(piece)
                # Stop once we have a full word: the next piece starts with a
                # space or punctuation, meaning the current word ended.
                if len(generated) > 1 and (piece.startswith(" ") or piece == ""):
                    generated.pop()
                    break
                current = torch.tensor([[next_id]])

            predicted = "".join(generated).strip().lower()
            # Compare only the leading alphanumeric run, matching the corpus's
            # own tokenization (_TOKEN_RE = [a-z0-9']+).
            predicted = "".join(c for c in predicted if c.isalnum() or c == "'")
            total_correct += int(predicted == true_word)
            total += 1
    return total_correct / max(total, 1), total


def trivial_baseline(task, sequences):
    """Accuracy of always predicting the corpus's single most frequent token."""
    from collections import Counter

    counts = Counter(
        token for tokens in task._author_examples.values() for t in tokens for token in t
    )
    most_common_token = counts.most_common(1)[0][0]
    total_correct = sum(
        sum(1 for word in words[1:] if word == most_common_token) for words in sequences
    )
    total = sum(len(words) - 1 for words in sequences)
    return most_common_token, total_correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--partitions", type=int, default=20)
    parser.add_argument("--sequences", type=int, default=200)
    parser.add_argument("--word-sequences", type=int, default=40)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading held-out authors (num_partitions={args.partitions})...", flush=True)
    sequences, task = load_heldout_word_sequences(args.partitions, args.sequences)
    print(f"{len(sequences)} held-out sequences", flush=True)

    token, trivial = trivial_baseline(task, sequences)
    print(f"trivial baseline (always '{token}'): {trivial:.4f}", flush=True)

    print(f"\nloading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    accuracy, count = bpe_token_accuracy(model, tokenizer, sequences)
    print(f"\n[1] BPE-token top-1 accuracy: {accuracy:.4f}  ({count:,} tokens)", flush=True)
    print("    -> over GPT-2's 50257 subword vocab. NOT comparable to the 15%.", flush=True)

    accuracy, count = word_level_accuracy(
        model, tokenizer, sequences[: args.word_sequences]
    )
    print(f"\n[2] Word-level top-1 accuracy: {accuracy:.4f}  ({count:,} words)", flush=True)
    print("    -> whole-word prediction. THIS is comparable to the 15%.", flush=True)


if __name__ == "__main__":
    main()
