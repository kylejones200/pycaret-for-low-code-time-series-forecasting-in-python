"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CrossValidationVisualizer:
    def __init__(self, data_length=200):
        np.random.seed(42)
        self.dates = pd.date_range("2023-01-01", periods=data_length)
        self.data = np.cumsum(np.random.randn(data_length)) + 20

    def plot_expanding_window(self, n_splits=3):
        """Visualize expanding window cross-validation"""
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(self.dates, self.data, "k-", alpha=0.2, label="Full Dataset")
        total_points = len(self.data)
        split_size = total_points // (n_splits + 1)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_splits))
        for i, color in enumerate(colors):
            train_end = split_size * (i + 1)
            test_start = train_end
            test_end = train_end + split_size
            if test_end > total_points:
                break
            ax.plot(
                self.dates[:train_end],
                self.data[:train_end],
                color=color,
                label=f"Train {i + 1}",
            )
            ax.plot(
                self.dates[test_start:test_end],
                self.data[test_start:test_end],
                "--",
                color=color,
                label=f"Test {i + 1}",
            )
            ax.axvspan(self.dates[0], self.dates[train_end - 1], alpha=0.1, color=color)
            ax.axvspan(
                self.dates[test_start], self.dates[test_end - 1], alpha=0.2, color=color
            )
        ax.set_title("Expanding Window Cross-Validation")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        return fig

    def plot_rolling_window(self, window_size=40, step_size=20):
        """Visualize rolling window cross-validation"""
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(self.dates, self.data, "k-", alpha=0.2, label="Full Dataset")
        total_points = len(self.data)
        n_splits = (total_points - window_size) // step_size
        colors = plt.cm.rainbow(np.linspace(0, 1, n_splits))
        for i, color in enumerate(colors):
            start = i * step_size
            train_end = start + window_size
            if train_end >= total_points:
                break
            ax.plot(
                self.dates[start:train_end],
                self.data[start:train_end],
                color=color,
                label=f"Train {i + 1}",
            )
            ax.axvspan(
                self.dates[start], self.dates[train_end - 1], alpha=0.1, color=color
            )
        ax.set_title("Rolling Window Cross-Validation")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        return fig

    def plot_sliding_window(self, window_size=40, horizon=10):
        """Visualize sliding window cross-validation"""
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(self.dates, self.data, "k-", alpha=0.2, label="Full Dataset")
        total_points = len(self.data)
        n_splits = (total_points - window_size - horizon) // horizon
        colors = plt.cm.rainbow(np.linspace(0, 1, max(1, n_splits)))
        for i, color in enumerate(colors):
            start = i * horizon
            train_end = start + window_size
            test_end = train_end + horizon
            if test_end >= total_points:
                break
            ax.plot(
                self.dates[start:train_end],
                self.data[start:train_end],
                color=color,
                label=f"Train {i + 1}",
            )
            ax.plot(
                self.dates[train_end:test_end],
                self.data[train_end:test_end],
                "--",
                color=color,
                label=f"Forecast {i + 1}",
            )
            ax.axvspan(
                self.dates[start], self.dates[train_end - 1], alpha=0.1, color=color
            )
            ax.axvspan(
                self.dates[train_end], self.dates[test_end - 1], alpha=0.2, color=color
            )
        ax.set_title("Sliding Window Cross-Validation")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        return fig
