"""Prepare and visualize sample data.

- load_data(path) -> (X, y)
- plot_raw(X, y, show, out_path)
- plot_scaled(X, y, show, out_path)
- example_hist(show, out_path)

The CLI allows saving plots to an output directory and running headless.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer

LOGGER = logging.getLogger(__name__)

def load_data(csv_path: Path = Path('samples/drawndata1.csv')) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    X = df[['x', 'y']].to_numpy()
    # normalize labels: strip whitespace and compare case-insensitively
    y = df['z'].astype(str).str.strip().str.lower() == 'a'
    return X, y.to_numpy()


def plot_scatter(X: np.ndarray, y: np.ndarray, *, xlabel: str = 'x', ylabel: str = 'y', show: bool = True, out_path: Path | None = None) -> None:
    """Plot a scatter of X[:,0] vs X[:,1] colored by boolean y.

    If out_path is provided, the figure is saved. If show is True, plt.show()
    will be called (requires a GUI backend).
    """
    plt.figure(figsize=(6, 5))
    # use explicit colors for clarity
    colors = np.where(y, 'tab:orange', 'tab:blue')
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        LOGGER.info('Saved scatter to %s', out_path)
    if show:
        plt.show()
    plt.close()

def plot_output_with_transformer(X: np.ndarray, y: np.ndarray, scaler: QuantileTransformer) -> None:
    """Plot the output of the transformer."""
    X_transformed = scaler.fit_transform(X)
    plot_scatter(X_transformed, y, xlabel='x (transformed)', ylabel='y (transformed)')

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Prepare and visualize sample data')
    p.add_argument('--csv', type=Path, default=Path('samples/drawndata1.csv'), help='Path to input CSV')
    p.add_argument('--out-dir', type=Path, default=Path('output/prepare-data'), help='Directory to save plots')
    p.add_argument('--show', action='store_true', help='Show plots with plt.show() (default: False)')
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    LOGGER.info('Loading data from %s', args.csv)
    X, y = load_data(args.csv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_scatter(X, y, show=args.show, out_path=out_dir / 'raw_scatter.png')
    plot_output_with_transformer(X, y, scaler=StandardScaler())
    plot_output_with_transformer(X, y, scaler=QuantileTransformer())

    return 0

if __name__ == '__main__':
    raise SystemExit(main())