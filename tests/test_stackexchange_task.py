"""Tests for the pure, network-free parts of the Stack Exchange task."""

import unittest

import torch

from pytorchexample.tasks.stackexchange import (
    PAD_ID,
    SEQ_LEN,
    StackExchangeLSTM,
    _clean_text,
    _select_partitions,
    _tokenize,
)


class CleanTextTest(unittest.TestCase):
    """Verify HTML stripping and tokenization."""

    def test_strips_tags_and_unescapes_entities(self) -> None:
        """Markup and HTML entities are removed, not part of the tokens."""
        raw = "<p>3D printing &amp; CAD are &quot;fun&quot;</p>"
        cleaned = _clean_text(raw)

        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertIn("CAD", cleaned)

    def test_tokenize_lowercases_and_splits_on_non_alnum(self) -> None:
        """Tokens are lowercase, alphanumeric-only words."""
        tokens = _tokenize("3D printing, CAD-models: fun!")
        self.assertEqual(tokens, ["3d", "printing", "cad", "models", "fun"])


class SelectPartitionsTest(unittest.TestCase):
    """Verify author ranking, eligibility filtering, and held-out split."""

    def _examples(self, counts: dict[int, int]) -> list[tuple[int, list[str]]]:
        """Build synthetic (author_id, tokens) pairs with the given answer counts."""
        return [
            (author_id, ["word"])
            for author_id, count in counts.items()
            for _ in range(count)
        ]

    def test_selects_most_active_authors_first(self) -> None:
        """Ranking is strictly by answer count, ties broken by author_id."""
        examples = self._examples({1: 5, 2: 10, 3: 3, 4: 8, 5: 2})
        author_to_partition, held_out = _select_partitions(
            examples, num_partitions=2, held_out_authors=1
        )

        # Most active: 2 (10), 4 (8), 1 (5) -> top 2 are {2, 4}, held-out is {1}.
        self.assertEqual(set(author_to_partition), {2, 4})
        self.assertEqual(held_out, [1])

    def test_partition_ids_are_dense_zero_based(self) -> None:
        """Partition ids assigned to selected authors are 0..num_partitions-1."""
        examples = self._examples({10: 5, 20: 4, 30: 3})
        author_to_partition, _ = _select_partitions(
            examples, num_partitions=2, held_out_authors=1
        )
        self.assertEqual(sorted(author_to_partition.values()), [0, 1])

    def test_ineligible_authors_are_excluded(self) -> None:
        """Authors below the minimum answer count never become clients."""
        examples = self._examples({1: 5, 2: 1})  # author 2 has only 1 answer
        author_to_partition, held_out = _select_partitions(
            examples,
            num_partitions=1,
            held_out_authors=0,
            min_answers_per_author=2,
        )
        self.assertEqual(set(author_to_partition), {1})
        self.assertNotIn(2, author_to_partition)
        self.assertNotIn(2, held_out)

    def test_raises_when_not_enough_qualifying_authors(self) -> None:
        """A clear error is raised instead of silently returning too few."""
        examples = self._examples({1: 5, 2: 5})
        with self.assertRaisesRegex(ValueError, "MAX_QUESTIONS_SCANNED"):
            _select_partitions(examples, num_partitions=5, held_out_authors=1)


class StackExchangeLSTMTest(unittest.TestCase):
    """Verify the model's forward pass shape and padding behavior."""

    def test_forward_output_shape(self) -> None:
        """Logits are produced for every position, over the full vocabulary."""
        model = StackExchangeLSTM(vocab_size=50, embed_dim=8, hidden_dim=16)
        batch = torch.randint(0, 50, (4, SEQ_LEN - 1))

        logits = model(batch)

        self.assertEqual(logits.shape, (4, SEQ_LEN - 1, 50))

    def test_embedding_wires_pad_id_as_padding_index(self) -> None:
        """The embedding must zero out PAD_ID so padding never influences the LSTM."""
        model = StackExchangeLSTM(vocab_size=50, embed_dim=8, hidden_dim=16)
        self.assertTrue(torch.all(model.embedding.weight[PAD_ID] == 0))


if __name__ == "__main__":
    unittest.main()
