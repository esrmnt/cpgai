from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)

def load_data(csv_path: Path = Path('samples/drawndata2.csv')) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    X = df[['x', 'y']].to_numpy()
    # normalize labels: strip whitespace and compare case-insensitively
    y = df['z'].astype(str).str.strip().str.lower() == 'a'
    return X, y.to_numpy()

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Prepare and visualize sample data')
    p.add_argument('--csv', type=Path, default=Path('samples/drawndata2.csv'), help='Path to input CSV')
    p.add_argument('--out-dir', type=Path, default=Path('output/prepare-data'), help='Directory to save plots')
    p.add_argument('--show', action='store_true', help='Show plots with plt.show() (default: False)')
    return p

def build_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('model', LogisticRegression())
    ])
    predictions = pipe.fit(X, y).predict(X)
    return predictions

def plot_scatter(X: np.ndarray, Pred: np.ndarray, xlabel: str = 'x', ylabel: str = 'y', show: bool = True, out_path: Path | None = None) -> None:
    """Plot a scatter of X[:,0] vs X[:,1] colored by boolean y."""
    plt.scatter(X[:, 0], X[:, 1], c=Pred)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        LOGGER.info('Saved scatter to %s', out_path)
    if show:
        plt.show()
    plt.close()

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    LOGGER.info('Loading data from %s', args.csv)
    X, y = load_data(args.csv)

    print(f'Data shape: X={X.shape}, y={y.shape}')

    predictions = build_model(X, y)
    plot_scatter(X, predictions, xlabel='x', ylabel='y', show=args.show, out_path=args.out_dir / 'polynomial-features-scatter.png')

    return 0

if __name__ == '__main__':
    raise SystemExit(main())