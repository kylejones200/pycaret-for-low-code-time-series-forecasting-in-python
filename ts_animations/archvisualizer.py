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

class ARCHVisualizer:

    def __init__(self):
        np.random.seed(42)
        self.n = 1000
        self.omega = 0.1
        self.alpha = 0.8
        self.generate_data()
        self.fit_models()

    def generate_data(self):
        errors = np.random.normal(size=self.n)
        self.volatility = np.zeros(self.n)
        self.returns = np.zeros(self.n)
        for t in range(1, self.n):
            self.volatility[t] = np.sqrt(self.omega + self.alpha * errors[t - 1] ** 2)
            self.returns[t] = self.volatility[t] * np.random.normal()
        self.data = pd.DataFrame({'returns': self.returns, 'volatility': self.volatility})

    def fit_models(self):
        self.model = arch_model(self.data['returns'], vol='ARCH', p=1)
        self.result = self.model.fit(disp='off')
        self.forecasts = []
        window = 100
        for i in range(window, self.n):
            train = self.returns[i - window:i]
            model = arch_model(train, vol='ARCH', p=1)
            result = model.fit(disp='off')
            forecast = result.forecast(horizon=10)
            self.forecasts.append(forecast.variance.iloc[-1])
        self.forecasts = np.array(self.forecasts)
        self.forecast_ylim = (0, np.percentile(self.forecasts, 99))

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        def animate(frame):
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            window = 100
            start_idx = frame % (self.n - window)
            end_idx = start_idx + window
            ax1.plot(self.data.index[start_idx:end_idx], self.returns[start_idx:end_idx], 'b-', label='Returns', alpha=0.7)
            ax1.plot(self.data.index[start_idx:end_idx], self.volatility[start_idx:end_idx], 'r-', label='True Volatility')
            ax1.fill_between(self.data.index[start_idx:end_idx], -self.volatility[start_idx:end_idx], self.volatility[start_idx:end_idx], color='r', alpha=0.2)
            ax1.set_title('Returns and Volatility')
            ax1.legend()
            ax1.grid(True)
            if start_idx >= window:
                forecast_idx = start_idx - window
                horizons = np.arange(1, 11)
                current_forecast = self.forecasts[forecast_idx]
                ax2.plot(horizons, current_forecast, 'g-', marker='o', label='Forecast')
                ax2.set_ylim(self.forecast_ylim)
                ax2.set_xlim(0.5, 10.5)
                ax2.set_title('10-step Volatility Forecast')
                ax2.set_xlabel('Forecast Horizon')
                ax2.set_ylabel('Predicted Variance')
                ax2.legend()
                ax2.grid(True)
                ax2.axhline(y=self.volatility[start_idx] ** 2, color='r', linestyle='--', label='Current Volatility')
            residuals = self.result.resid[start_idx:end_idx]
            ax3.hist(residuals, bins=30, density=True, alpha=0.7, range=(-4, 4))
            ax3.set_xlim(-4, 4)
            ax3.set_title('Standardized Residuals Distribution')
            ax3.grid(True)
            plt.tight_layout()
        anim = FuncAnimation(fig, animate, frames=200, interval=100, repeat=True)
        return anim

