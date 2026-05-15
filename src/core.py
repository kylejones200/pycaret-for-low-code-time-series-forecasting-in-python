"""Core functions for PyCaret low-code time series forecasting."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def prepare_time_series_data(
    df: pd.DataFrame, date_col: str, value_col: str
) -> pd.Series:
    """Prepare time series data for PyCaret."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    return df[value_col]


def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate forecast error metrics."""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def plot_pycaret_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    output_path: Path,
    plot: bool = False,
):
    """Plot PyCaret forecast"""
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(actual, label="Actual", color="#4A90A4", linewidth=1.2)
        ax.plot(predicted, label="Predicted", color="#D4A574", linewidth=1.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="best")

        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
