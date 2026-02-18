"""
Shared utility functions used across the readmission prediction pipeline.

Author: Ronald Wen
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create a directory (and any parents) if it does not already exist.

    Author: Ronald Wen
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: Path) -> None:
    """Serialise a dictionary to a JSON file with readable indentation.

    Author: Ronald Wen
    """
    ensure_dir(path.parent)
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return its contents as a Python dict.

    Author: Ronald Wen
    """
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def summarise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of dtype, null count, and cardinality per column.

    Useful for a quick audit of a raw or processed feature matrix.

    Author: Ronald Wen
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'null_count': df.isna().sum(),
        'null_pct': (df.isna().sum() / len(df) * 100).round(2),
        'n_unique': df.nunique(),
    })
    return summary.sort_values('null_pct', ascending=False)


def check_class_balance(y: pd.Series) -> None:
    """Print class distribution statistics for a binary target series.

    Author: Ronald Wen
    """
    counts = y.value_counts()
    total = len(y)
    print("Class distribution:")
    for cls, cnt in counts.items():
        print(f"  Class {cls}: {cnt:,}  ({cnt / total * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str = 'Block'):
    """Context manager that prints wall-clock time for a code block.

    Usage:
        with timer("Training XGBoost"):
            model.fit(X_train, y_train)

    Author: Ronald Wen
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label} completed in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def set_plot_style() -> None:
    """Apply a clean, publication-ready matplotlib style.

    Author: Ronald Wen
    """
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk and close it to free memory.

    Author: Ronald Wen
    """
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {path}")


# ---------------------------------------------------------------------------
# Metrics formatting
# ---------------------------------------------------------------------------

def metrics_to_dataframe(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert the nested metrics dict (model → metric → value) to a DataFrame.

    Author: Ronald Wen
    """
    rows = []
    for model_name, model_metrics in metrics.items():
        row = {'Model': model_name}
        row.update(model_metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Model')
    return df
