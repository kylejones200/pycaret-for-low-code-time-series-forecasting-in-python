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

class TimeSeriesFeatureVisualizer:

    def __init__(self):
        np.random.seed(42)
        n = 500
        self.dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        self.values = 100 + np.cumsum(np.random.normal(0, 1, n))
        self.df = pd.DataFrame({'date': self.dates, 'value': self.values})
        self.calculate_features()

    def calculate_features(self):
        self.df['rolling_mean_7'] = self.df['value'].rolling(window=7).mean()
        self.df['rolling_std_7'] = self.df['value'].rolling(window=7).std()
        self.df['lag_1'] = self.df['value'].shift(1)
        self.df['lag_7'] = self.df['value'].shift(7)
        t = np.arange(len(self.df))
        period = 30
        self.df['fourier_sin'] = np.sin(2 * np.pi * t / period)
        self.df['fourier_cos'] = np.cos(2 * np.pi * t / period)

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        self.window = 100

        def animate(frame):
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            start_idx = max(0, frame - self.window)
            end_idx = frame + 1
            ax1.plot(self.df['date'][start_idx:end_idx], self.df['value'][start_idx:end_idx], label='Original', color='blue')
            ax1.plot(self.df['date'][start_idx:end_idx], self.df['rolling_mean_7'][start_idx:end_idx], label='7-day MA', color='red')
            ax1.fill_between(self.df['date'][start_idx:end_idx], self.df['rolling_mean_7'][start_idx:end_idx] - self.df['rolling_std_7'][start_idx:end_idx], self.df['rolling_mean_7'][start_idx:end_idx] + self.df['rolling_std_7'][start_idx:end_idx], alpha=0.2, color='red')
            ax1.set_title('Time Series with Rolling Statistics')
            ax1.legend()
            if frame > 7:
                ax2.scatter(self.df['lag_1'][start_idx:end_idx], self.df['value'][start_idx:end_idx], alpha=0.5, label='Lag 1')
                ax2.scatter(self.df['lag_7'][start_idx:end_idx], self.df['value'][start_idx:end_idx], alpha=0.5, label='Lag 7')
                ax2.set_title('Lagged Features')
                ax2.set_xlabel('Lagged Value')
                ax2.set_ylabel('Current Value')
                ax2.legend()
            ax3.plot(self.df['date'][start_idx:end_idx], self.df['fourier_sin'][start_idx:end_idx], label='Sine', color='green')
            ax3.plot(self.df['date'][start_idx:end_idx], self.df['fourier_cos'][start_idx:end_idx], label='Cosine', color='orange')
            ax3.set_title('Fourier Components')
            ax3.legend()
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=len(self.df) - self.window, interval=200, repeat=True)
        return anim

