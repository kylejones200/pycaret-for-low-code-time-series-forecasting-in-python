"""Auto-split from legacy monolithic script."""

from arch import arch_model
from datetime import datetime
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from pmdarima import auto_arima
from pycaret.time_series import TSForecastingExperiment
from pycaret.time_series import setup, compare_models, tune_model, predict_model, plot_model, pull, save_model, load_model
from pycaret.time_series import stack_models
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
import arviz as az
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

class AnimatedCrossValidation:

    def __init__(self, data_length=200):
        np.random.seed(42)
        self.dates = pd.date_range('2023-01-01', periods=data_length)
        self.data = np.cumsum(np.random.randn(data_length)) + 20

    def create_expanding_window_animation(self, n_splits=3):
        fig, ax = plt.subplots(figsize=(12, 6))
        total_points = len(self.data)
        split_size = total_points // (n_splits + 1)

        def animate(i):
            ax.clear()
            ax.plot(self.dates, self.data, 'k-', alpha=0.2, label='Full Dataset')
            train_end = split_size * (i // 15 + 1)
            test_start = train_end
            test_end = train_end + split_size
            if test_end <= total_points:
                ax.plot(self.dates[:train_end], self.data[:train_end], 'b-', label='Training')
                ax.plot(self.dates[test_start:test_end], self.data[test_start:test_end], 'r--', label='Test')
                ax.axvspan(self.dates[0], self.dates[train_end - 1], alpha=0.1, color='blue')
                ax.axvspan(self.dates[test_start], self.dates[test_end - 1], alpha=0.2, color='red')
            ax.set_title('Expanding Window Cross-Validation')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=15 * n_splits, interval=500, repeat=True)
        return anim

    def create_rolling_window_animation(self, window_size=40, step_size=20):
        fig, ax = plt.subplots(figsize=(12, 6))
        total_points = len(self.data)
        n_splits = (total_points - window_size) // step_size

        def animate(i):
            ax.clear()
            ax.plot(self.dates, self.data, 'k-', alpha=0.2, label='Full Dataset')
            start = i // 15 * step_size
            train_end = start + window_size
            if train_end <= total_points:
                ax.plot(self.dates[start:train_end], self.data[start:train_end], 'b-', label='Training Window')
                ax.axvspan(self.dates[start], self.dates[train_end - 1], alpha=0.2, color='blue')
            ax.set_title('Rolling Window Cross-Validation')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=15 * n_splits, interval=500, repeat=True)
        return anim

    def create_sliding_window_animation(self, window_size=40, horizon=10):
        fig, ax = plt.subplots(figsize=(12, 6))
        total_points = len(self.data)
        n_splits = (total_points - window_size - horizon) // horizon

        def animate(i):
            ax.clear()
            ax.plot(self.dates, self.data, 'k-', alpha=0.2, label='Full Dataset')
            start = i // 15 * horizon
            train_end = start + window_size
            test_end = train_end + horizon
            if test_end <= total_points:
                ax.plot(self.dates[start:train_end], self.data[start:train_end], 'b-', label='Training')
                ax.plot(self.dates[train_end:test_end], self.data[train_end:test_end], 'r--', label='Forecast')
                ax.axvspan(self.dates[start], self.dates[train_end - 1], alpha=0.1, color='blue')
                ax.axvspan(self.dates[train_end], self.dates[test_end - 1], alpha=0.2, color='red')
            ax.set_title('Sliding Window Cross-Validation')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=15 * n_splits, interval=500, repeat=True)
        return anim

