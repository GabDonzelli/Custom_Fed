"""Utility to log and save FL training results to CSV."""

import csv
from datetime import datetime
from pathlib import Path


class ResultsLogger:
    """Log FL results (metrics per round) to a CSV file."""

    def __init__(self, filename: str = None) -> None:
        """Initialize logger with filename (auto-generated with timestamp).

        Args:
            filename: Output CSV filename. If None, auto-generates with timestamp.
                     Example: fl_results_2024_09_01_103045.csv
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            filename = f"fl_results_{timestamp}.csv"

        self.filename = Path(filename)
        self.csv_file = None
        self.writer = None
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        """Create CSV file with headers."""
        self.csv_file = open(self.filename, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timestamp",
                "round",
                "accuracy",
                "loss",
                "num_clients_trained",
            ],
        )
        self.writer.writeheader()
        self.csv_file.flush()
        print(f"📊 Logging results to: {self.filename.absolute()}")

    def log_round(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        num_clients: int = None,
    ) -> None:
        """Log metrics for one FL round.

        Args:
            round_num: Round number (1-based)
            accuracy: Model accuracy on test set
            loss: Model loss on test set
            num_clients: Number of clients trained in this round
        """
        row = {
            "timestamp": datetime.now().isoformat(),
            "round": round_num,
            "accuracy": f"{accuracy:.6f}",
            "loss": f"{loss:.6f}",
            "num_clients_trained": num_clients or "N/A",
        }
        self.writer.writerow(row)
        self.csv_file.flush()
        print(
            f"✓ Round {round_num}: accuracy={accuracy:.4f}, loss={loss:.4f}"
        )

    def close(self) -> None:
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()
            print(f"✓ Results saved to {self.filename}")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, *args):
        """Context manager support."""
        self.close()
