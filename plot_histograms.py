"""Build reproducible histograms for Lab 1 NumPy matrix statistics."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics

ROOT = Path(__file__).resolve().parent

def save_histograms():
    """Draw histograms for each row and column of the seeded random matrix."""
    data = RandomMatrixInput(rows=20, columns=12, mean=0.0, std=1.0, seed=42)
    matrix = matrix_statistics(data).matrix
    for axis, vectors in (("rows", matrix), ("columns", matrix.T)):
        fig, ax = plt.subplots(figsize=(9, 5))
        for index, values in enumerate(vectors):
            ax.hist(values, bins=12, alpha=0.26, label=f"{axis[:-1]} {index + 1}")
        ax.set_title(f"Distribution of values by {axis} (seed=42)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        if len(vectors) <= 12:
            ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(ROOT / f"histogram_{axis}.png", dpi=150)
        plt.close(fig)
    print("Saved histogram_rows.png and histogram_columns.png")

if __name__ == "__main__":
    save_histograms()
