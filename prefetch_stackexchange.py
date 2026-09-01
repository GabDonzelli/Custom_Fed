"""One-time helper: scan and cache Stack Exchange data before running offline.

Run this once on a machine with internet access (e.g. a cluster's login
node) before submitting jobs that use `task-name = "stackexchange"` on
machines without internet access (e.g. SLURM compute nodes). It populates
`pytorchexample/tasks/.stackexchange_cache/`, which later runs reuse instead
of streaming from the Hugging Face Hub again.

Usage:
    python prefetch_stackexchange.py --num-partitions 100
"""

import argparse

from pytorchexample.tasks.stackexchange import StackExchangeTask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-partitions",
        type=int,
        required=True,
        help="Same value as num-partitions in pyproject.toml.",
    )
    args = parser.parse_args()

    task = StackExchangeTask()
    task.prefetch(num_partitions=args.num_partitions)
    print(
        f"Cached {len(task._author_examples)} training partitions and "
        f"{len(task._held_out_tokens)} held-out examples."
    )


if __name__ == "__main__":
    main()
