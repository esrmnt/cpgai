""" Linear regression with sample Iris dataset.

This module provides functions to load the iris dataset, train a simple linear regression model to predict petal width from petal length, evaluate it, and produce plots. It is written to be usable both as a script (CLI) and as an importable module for tests.

Usage examples:
    # Run and show plots (requires GUI backend)
    python iris.py --show

    # Run headless and save plots to ./output
    python iris.py --out-dir output
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """Load the iris dataset and return a DataFrame.

    Returns:
        DataFrame: iris features with column names from the dataset.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    return df


def prepare_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare X and y arrays for regression: predict petal width from petal length.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) shaped (n_samples, 1)
    """
    y = df['petal width (cm)'].values.reshape(-1, 1)
    X = df['petal length (cm)'].values.reshape(-1, 1)
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int | None = 42):
    """Split the data, train a LinearRegression model and return the results.

    Returns:
        dict: contains train/test splits, trained model, and predictions on test set.
    """
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    return {
        'model': model,
        'xtrain': xtrain,
        'xtest': xtest,
        'ytrain': ytrain,
        'ytest': ytest,
        'y_pred': y_pred,
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute common regression metrics.

    Returns:
        dict: mae, mse, rmse
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'mae': mae, 'mse': mse, 'rmse': rmse}


def plot_actual_vs_pred(xtest: np.ndarray, ytest: np.ndarray, y_pred: np.ndarray, *, show: bool = True, out_path: Path | None = None) -> None:
    """Scatter plot of actual vs predicted petal widths.

    If out_path is provided the plot will be saved there. If show is True,
    plt.show() will be called (this requires a GUI-capable backend).
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(xtest, ytest, color='blue', label='Actual')
    plt.scatter(xtest, y_pred, color='red', label='Predicted')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Actual vs Predicted Petal Width')
    plt.legend()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        LOGGER.info('Saved actual vs predicted plot to %s', out_path)
    if show:
        plt.show()
    plt.close()


def plot_sepal_scatter(df: pd.DataFrame, *, show: bool = True, out_path: Path | None = None) -> None:
    """Scatter plot of sepal length vs sepal width.

    Same saving/showing behavior as plot_actual_vs_pred.
    """
    ax = df.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', title='Iris Dataset Scatter Plot')
    fig = ax.get_figure()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        LOGGER.info('Saved sepal scatter plot to %s', out_path)
    if show:
        plt.show()
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Iris linear regression demo (predict petal width from petal length)')
    parser.add_argument('--show', dest='show', action='store_true', help='Call plt.show() to display plots (default: False)')
    parser.add_argument('--out-dir', type=Path, default=Path('output'), help='Directory to save plots (if provided)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for train/test split')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    LOGGER.info('Loading data')
    df = load_data()
    LOGGER.info('Columns: %s', list(df.columns))
    LOGGER.info('\n%s', df.describe())

    X, y = prepare_xy(df)
    result = train_model(X, y, random_state=args.random_state)

    model = result['model']
    LOGGER.info('Model intercept: %s coef: %s', model.intercept_, model.coef_)
    LOGGER.info('Predicted for 1.5 cm petal length: %s', model.predict([[1.5]]))

    metrics = evaluate(result['ytest'], result['y_pred'])
    LOGGER.info('Evaluation metrics: %s', metrics)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save plots if out_dir provided
    avp_path = out_dir / 'actual_vs_predicted.png'
    plot_actual_vs_pred(result['xtest'], result['ytest'], result['y_pred'], show=args.show, out_path=avp_path)

    sepal_path = out_dir / 'sepal_scatter.png'
    plot_sepal_scatter(df, show=args.show, out_path=sepal_path)

    # Print metric values to stdout for quick visibility
    print('Mean Absolute Error:', metrics['mae'])
    print('Mean Squared Error:', metrics['mse'])
    print('Root Mean Squared Error:', metrics['rmse'])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())